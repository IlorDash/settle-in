"""Score both intent classifiers on the test set - measurement group B.

The local classifier and the language model are given the same 450 messages,
one at a time and with no conversation behind them, which is the boundary
section 4.5 draws: what is measured is the label a classifier returns for a
message on its own, not what the deployed bot finally does with it.

Three things this script has to get right, and each one has a reason:

  * It calls the chain the bot calls. `_build_classifier_chain` is private,
    and importing it is deliberate - re-typing CLASSIFICATION_PROMPT here
    would measure a classifier the bot does not run, and would drift silently
    the moment that prompt is edited.
  * Every label it buys is written to disk before the next call. The threshold
    sweep needs a language-model label for each message below the threshold,
    and those are the same labels at every threshold, since the model runs at
    temperature zero on one message with no history. Without the cache the
    sweep would re-buy them at each of eleven thresholds.
  * Token counts come from a callback rather than from changing the chain.
    The chain ends in StrOutputParser, which drops usage_metadata, but a
    handler passed at invoke time sees the model's reply before the parser.

Run from the repo root:

    python -m scripts.evaluate_system --local-only     # free
    python -m scripts.evaluate_system --limit 20       # a few cents
    python -m scripts.evaluate_system                  # the whole set

Re-running is cheap once the cache is warm: only messages it has never seen
cost anything.
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from sklearn.metrics import classification_report, confusion_matrix

from src.agents.intent_classifier import classify, load_classifier
from src.agents.orchestrator import (
    CONFIDENCE_THRESHOLD,
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
    LLM_MODEL,
    _build_classifier_chain,
)

TEST_SET_PATH = Path("data/intent_test.csv")
CACHE_PATH = Path("data/eval_cache_llm_intent.jsonl")
REPORT_PATH = Path("data/eval_routing.json")

INTENTS = (INTENT_KNOWLEDGE_QUESTION, INTENT_TRANSLATION, INTENT_OUT_OF_SCOPE)

# Published pricing per million tokens (OpenAI, 2026). The local classifier
# has no such figure: it runs on the machine that is already running the bot.
PRICE_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # The translation agent runs on the larger model, so anything pricing a
    # whole conversation has to know both. Group B never calls it.
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# The sweep runs at every tenth, and CONFIDENCE_THRESHOLD is added so the
# deployed setting appears in the table whatever its value.
SWEEP_STEPS = 11


@dataclass(frozen=True)
class Example:
    """One labelled test message.

    Attributes:
        text: The message as a user would send it.
        intent: The label a classifier ought to return for it.
        language: The language it is written in, for the per-language report.
    """

    text: str
    intent: str
    language: str


@dataclass(frozen=True)
class Prediction:
    """What one classifier returned for one message.

    Attributes:
        label: The intent it chose.
        confidence: Its own probability, or None for the language model,
            which reports none.
        latency_ms: Wall-clock time for that single classification.
        input_tokens: Prompt tokens billed, or 0 when nothing was bought.
        output_tokens: Completion tokens billed, or 0.
    """

    label: str
    confidence: float | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class UsageRecorder(BaseCallbackHandler):
    """Catches the model's token counts before the output parser drops them."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        """Read usage off the raw message the model returned."""
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)


@dataclass
class LabelCache:
    """Every language-model label bought so far, keyed by the message.

    Attributes:
        path: The JSONL file it is written to.
        entries: What has been read back from that file.

    A record is appended as soon as its call returns, rather than at the end
    of the run, because a crash halfway through must not discard calls that
    have already been paid for.
    """

    path: Path
    entries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path) -> "LabelCache":
        """Load whatever the file already holds."""
        cache = cls(path=path)
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    cache.entries[record["text"]] = record
        return cache

    def get(self, text: str) -> dict | None:
        """Return the record for a message, or None if it was never bought."""
        return self.entries.get(text)

    def put(self, record: dict) -> None:
        """Store a record and flush it to disk immediately."""
        self.entries[record["text"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_test_set(path: Path) -> list[Example]:
    """Read the labelled test messages.

    Args:
        path: The CSV written by scripts/generate_intent_test_set.py.

    Returns:
        Every row of it, in file order.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        return [
            Example(row["text"], row["intent"], row.get("language", ""))
            for row in csv.DictReader(handle)
        ]


def parse_llm_intent(reply: str) -> str:
    """Turn the model's reply into an intent, the way the orchestrator does.

    This mirrors the four lines that follow the escalation call in
    `classify_intent`. It is duplicated rather than imported because the
    orchestrator keeps that logic inside the graph node, and the two have to
    be kept in step by hand until it is lifted out.
    """
    lowered = reply.strip().lower()
    if INTENT_TRANSLATION in lowered:
        return INTENT_TRANSLATION
    if INTENT_OUT_OF_SCOPE in lowered:
        return INTENT_OUT_OF_SCOPE
    return INTENT_KNOWLEDGE_QUESTION


def classify_locally(classifier, examples: list[Example]) -> list[Prediction]:
    """Run the trained pipeline over every message, timing each one.

    Args:
        classifier: The pipeline returned by load_classifier().
        examples: The messages to classify.

    Returns:
        One prediction per message, in the same order.
    """
    predictions = []
    for example in examples:
        started = time.perf_counter()
        label, confidence = classify(classifier, example.text)
        elapsed = (time.perf_counter() - started) * 1000
        predictions.append(Prediction(label, confidence, elapsed))
    return predictions


async def buy_one_label(chain, text: str) -> dict:
    """Send one message to the language model and record what it cost.

    Args:
        chain: The escalation chain the orchestrator builds.
        text: The message to classify.

    Returns:
        A cache record: the message, the label, the raw reply, the latency and
        the token counts of the call that produced it.
    """
    recorder = UsageRecorder()
    started = time.perf_counter()
    reply = await chain.ainvoke(
        {"input": text, "history": []}, config={"callbacks": [recorder]}
    )
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "text": text,
        "label": parse_llm_intent(reply),
        "reply": reply.strip(),
        "latency_ms": round(elapsed, 1),
        "input_tokens": recorder.input_tokens,
        "output_tokens": recorder.output_tokens,
        "model": LLM_MODEL,
    }


async def classify_with_llm(examples: list[Example], cache: LabelCache) -> list[dict]:
    """Fetch a language-model label for every message, buying only new ones.

    Args:
        examples: The messages to classify.
        cache: Labels already bought, which are reused rather than re-bought.

    Returns:
        One cache record per message, in the same order.
    """
    chain = _build_classifier_chain()
    records, bought = [], 0
    for position, example in enumerate(examples, start=1):
        record = cache.get(example.text)
        if record is None:
            record = await buy_one_label(chain, example.text)
            cache.put(record)
            bought += 1
        records.append(record)
        if position % 50 == 0:
            print(f"  {position}/{len(examples)} done, {bought} bought")
    print(f"  {len(examples)} done, {bought} bought, {len(examples) - bought} cached")
    return records


def latency_summary(values: list[float]) -> dict:
    """Return the mean and the 95th percentile of a list of timings."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    return {
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p95_ms": round(ordered[index], 2),
        "n": len(ordered),
    }


def score(truths: list[str], predicted: list[str]) -> dict:
    """Accuracy, per-class precision, recall and F1, and the confusion matrix."""
    report = classification_report(
        truths, predicted, labels=list(INTENTS), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(truths, predicted, labels=list(INTENTS))
    return {
        "accuracy": round(report["accuracy"], 4),
        "per_class": {
            intent: {
                "precision": round(report[intent]["precision"], 4),
                "recall": round(report[intent]["recall"], 4),
                "f1": round(report[intent]["f1-score"], 4),
                "support": int(report[intent]["support"]),
            }
            for intent in INTENTS
        },
        "confusion": {
            actual: dict(zip(INTENTS, (int(n) for n in row)))
            for actual, row in zip(INTENTS, matrix)
        },
    }


def score_by_language(examples: list[Example], predicted: list[str]) -> dict:
    """Accuracy within each language of the test set."""
    languages = sorted({example.language for example in examples if example.language})
    result = {}
    for language in languages:
        pairs = [
            (example.intent, label)
            for example, label in zip(examples, predicted)
            if example.language == language
        ]
        hits = sum(1 for truth, label in pairs if truth == label)
        result[language] = {"accuracy": round(hits / len(pairs), 4), "n": len(pairs)}
    return result


def sweep_thresholds(
    examples: list[Example], local: list[Prediction], llm: list[dict]
) -> list[dict]:
    """Score the hybrid rule at every threshold, using labels already bought.

    Args:
        examples: The test messages, for their true labels.
        local: What the local classifier returned, with its confidences.
        llm: The cached language-model records.

    Returns:
        One row per threshold: the share escalated and the accuracy reached.

    The rule swept here is the threshold alone. The deployed bot has a second
    reason to escalate, described in section 4.3, which no message in this set
    can trigger, so the escalated share below is a floor for the deployed bot
    rather than its figure.
    """
    steps = [round(i / (SWEEP_STEPS - 1), 2) for i in range(SWEEP_STEPS)]
    if CONFIDENCE_THRESHOLD not in steps:
        steps = sorted(steps + [CONFIDENCE_THRESHOLD])
    rows = []
    for threshold in steps:
        labels, escalated = [], 0
        for prediction, record in zip(local, llm):
            if prediction.confidence is not None and prediction.confidence >= threshold:
                labels.append(prediction.label)
            else:
                labels.append(record["label"])
                escalated += 1
        hits = sum(1 for e, label in zip(examples, labels) if e.intent == label)
        rows.append(
            {
                "threshold": threshold,
                "escalated": escalated,
                "escalated_share": round(escalated / len(examples), 4),
                "accuracy": round(hits / len(examples), 4),
                "deployed": threshold == CONFIDENCE_THRESHOLD,
            }
        )
    return rows


def cost_per_thousand(records: list[dict]) -> dict:
    """Scale the tokens actually spent up to 1000 classifications."""
    price = PRICE_PER_MILLION.get(LLM_MODEL)
    inputs = sum(r["input_tokens"] for r in records)
    outputs = sum(r["output_tokens"] for r in records)
    if not records or price is None:
        return {"model": LLM_MODEL, "priced": False}
    spent = (inputs * price["input"] + outputs * price["output"]) / 1_000_000
    return {
        "model": LLM_MODEL,
        "priced": True,
        "input_tokens_mean": round(inputs / len(records), 1),
        "output_tokens_mean": round(outputs / len(records), 1),
        "usd_per_1000": round(spent / len(records) * 1000, 4),
        "usd_for_this_run": round(spent, 4),
    }


def print_report(result: dict) -> None:
    """Print the figures a person reads, in the order group B lists them."""
    print("\n" + "=" * 70)
    print(f"routing evaluation, {result['n']} messages")
    print("=" * 70)
    for name in ("local", "llm"):
        block = result.get(name)
        if not block:
            continue
        print(f"\n[{name}] accuracy {block['accuracy']:.4f}")
        for intent, row in block["per_class"].items():
            print(
                f"  {intent:<20} P {row['precision']:.3f}"
                f"  R {row['recall']:.3f}  F1 {row['f1']:.3f}"
                f"  n {row['support']}"
            )
        print("  confusion (rows are true labels):")
        for actual, row in block["confusion"].items():
            counts = " ".join(f"{predicted}={n}" for predicted, n in row.items())
            print(f"    {actual:<20} {counts}")
        print(f"  by language: {block['by_language']}")
        print(f"  latency: {block['latency']}")
    if result.get("cost"):
        print(f"\n[cost] {result['cost']}")
    if result.get("sweep"):
        print("\n[threshold sweep] escalation buys accuracy")
        print("  threshold  escalated  accuracy")
        for row in result["sweep"]:
            mark = "  <- deployed" if row["deployed"] else ""
            print(
                f"  {row['threshold']:>9}  {row['escalated_share']:>9.3f}"
                f"  {row['accuracy']:>8.4f}{mark}"
            )


async def run(examples: list[Example], local_only: bool) -> dict:
    """Measure both classifiers and assemble the report."""
    truths = [example.intent for example in examples]
    print(f"local classifier over {len(examples)} messages")
    local = classify_locally(load_classifier(), examples)
    local_labels = [prediction.label for prediction in local]
    result = {
        "n": len(examples),
        "local": score(truths, local_labels)
        | {
            "by_language": score_by_language(examples, local_labels),
            "latency": latency_summary([p.latency_ms for p in local]),
        },
    }
    if local_only:
        return result

    print(f"language model over {len(examples)} messages")
    cache = LabelCache.open(CACHE_PATH)
    records = await classify_with_llm(examples, cache)
    llm_labels = [record["label"] for record in records]
    result["llm"] = score(truths, llm_labels) | {
        "by_language": score_by_language(examples, llm_labels),
        "latency": latency_summary([record["latency_ms"] for record in records]),
    }
    result["cost"] = cost_per_thousand(records)
    result["sweep"] = sweep_thresholds(examples, local, records)
    return result


def main() -> None:
    """Parse the arguments, run the measurement, print and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="use only the first N messages")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="skip the language model, so the run is free",
    )
    arguments = parser.parse_args()

    examples = load_test_set(TEST_SET_PATH)
    if arguments.limit:
        examples = examples[: arguments.limit]

    result = asyncio.run(run(examples, arguments.local_only))
    print_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {REPORT_PATH.resolve()}")
    if not arguments.local_only:
        print(f"Labels bought are cached in {CACHE_PATH}")


if __name__ == "__main__":
    main()
