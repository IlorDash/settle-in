"""Time what a user waits for, per path - measurement group A.

Section 4.5 asks for the wait a user actually sees, broken down by stage, over
at least 30 messages per path. This script sends messages through the bot's
own orchestrator - the same graph, the same chains, the same prompts - and
times them from the outside, the way the user experiences it.

The three text paths are measured. A photograph is group E's subject and needs
an image, so the document path is not run here.

  * knowledge_question - classification, retrieval, generation
  * translation        - classification, generation
  * out_of_scope       - classification, then a canned reply and no model call

Four things this script has to get right:

  * It runs the deployed graph. `build_orchestrator` with the real RAG and
    translation chains is what the bot builds in its own post_init hook, so
    a figure here is a figure about the shipped system.
  * No checkpointer, so every message is a first message. That is the
    measured case: with a conversation behind it the knowledge chain buys an
    extra call to rewrite the question, which is named in the report and not
    averaged into these numbers.
  * The stage breakdown comes from callbacks, not from timers planted in
    `src/`. A handler sees each retriever span and each model call, and the
    remainder is reported as its own line instead of being distributed over
    the stages.
  * Whether the classifier escalates is known before the message is sent, by
    running the local classifier separately - it is deterministic - so the
    first model call of a turn can be attributed to classification or to the
    answer without guessing from prompt text.

Run from the repo root:

    python -m scripts.evaluate_latency --per-path 1      # a smoke test
    python -m scripts.evaluate_latency                   # A1, A2 and A4
    python -m scripts.evaluate_latency --concurrent      # adds A3

Give `--label` the name of the machine, since that is what the figures are
about: `--label "raspberry pi 4, 4gb"`.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from scripts.evaluate_system import PRICE_PER_MILLION
from src.agents.intent_classifier import classify, load_classifier
from src.agents.orchestrator import (
    CONFIDENCE_THRESHOLD,
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
    LLM_MODEL,
    build_orchestrator,
    process_message,
)
from src.agents.rag_agent import build_rag_chain
from src.agents.translation_agent import build_translation_chain
from src.knowledge.vectorstore import get_retriever, load_vectorstore

TEST_SET_PATH = Path("data/intent_test.csv")
REPORT_PATH = Path("data/eval_latency.json")
# Every measured turn is written here the moment it finishes. A run buys its
# answers and cannot get them back, so nothing downstream of the sending -
# summarising, pricing, printing - may be able to discard them by failing.
# `--from-turns` rebuilds the report out of this file and sends nothing.
TURNS_PATH = Path("data/eval_latency_turns.jsonl")

PATHS = (INTENT_KNOWLEDGE_QUESTION, INTENT_TRANSLATION, INTENT_OUT_OF_SCOPE)
DEFAULT_PER_PATH = 30
# Section 4.5 asks what queuing does to the percentile, and five is the number
# it names.
BURST_SIZE = 5


@dataclass
class Spans(BaseCallbackHandler):
    """Times every retriever search and every model call of one turn.

    Attributes:
        retrieval_ms: Total time inside the retriever.
        model_ms: One entry per model call, in the order they finished.
        input_tokens: Prompt tokens billed across the turn.
        output_tokens: Completion tokens billed across the turn.
    """

    retrieval_ms: float = 0.0
    model_ms: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # Keyed by model, because the agents do not all run on the same one:
    # the translation agent is on gpt-4o while everything else is on
    # gpt-4o-mini, and one price over both would be wrong by an order of
    # magnitude.
    tokens_by_model: dict = field(default_factory=dict)
    _retriever_started: float | None = field(default=None, init=False)
    _model_started: list[float] = field(default_factory=list, init=False)
    _model_names: list[str] = field(default_factory=list, init=False)

    def on_retriever_start(self, serialized, query, **kwargs) -> None:
        """Mark the start of a search."""
        self._retriever_started = time.perf_counter()

    def on_retriever_end(self, documents, **kwargs) -> None:
        """Add the search that just finished to the total."""
        if self._retriever_started is not None:
            self.retrieval_ms += (time.perf_counter() - self._retriever_started) * 1000
            self._retriever_started = None

    def _begin(self, **kwargs) -> None:
        """Record the start of a model call and which model it went to."""
        self._model_started.append(time.perf_counter())
        parameters = kwargs.get("invocation_params") or {}
        self._model_names.append(parameters.get("model") or "")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """Mark the start of a model call."""
        self._begin(**kwargs)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        """Same, for the chat interface the agents actually use."""
        self._begin(**kwargs)

    def on_llm_end(self, response, **kwargs) -> None:
        """Close the model call and read its token counts off the reply."""
        if self._model_started:
            started = self._model_started.pop()
            self.model_ms.append((time.perf_counter() - started) * 1000)
        name = self._model_names.pop() if self._model_names else ""
        # The reply names the model it came from, which beats the request
        # parameters when both are there.
        name = (response.llm_output or {}).get("model_name") or name
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        counted = self.tokens_by_model.setdefault(name, {"input": 0, "output": 0})
        counted["input"] += prompt_tokens
        counted["output"] += completion_tokens


@dataclass(frozen=True)
class Turn:
    """One measured message.

    Attributes:
        text: What was sent.
        expected_path: The path it was chosen to exercise.
        intent: The intent the orchestrator actually settled on.
        language: The language it is written in.
        total_ms: What the user waited, start to reply.
        local_ms: The local classifier, timed by a call of its own outside
            the graph. A proxy for the classification the graph does, not a
            slice of total_ms - which is why it is reported beside the
            breakdown and not inside it.
        escalated: Whether the classifier asked the language model.
        escalation_ms: The classification call to the model, or 0.0.
        retrieval_ms: Time inside the retriever.
        generation_ms: The model call that produced the answer, or 0.0.
        input_tokens: Prompt tokens billed for the whole turn.
        output_tokens: Completion tokens billed for the whole turn.
        tokens_by_model: The same counts split by the model that billed
            them, since the turn can touch two models at different prices.
    """

    text: str
    expected_path: str
    intent: str
    language: str
    total_ms: float
    local_ms: float
    escalated: bool
    escalation_ms: float
    retrieval_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int
    tokens_by_model: dict

    @property
    def other_ms(self) -> float:
        """What the three measured spans leave over.

        The local classifier, LangGraph's own dispatch between nodes, and
        the state merging around them. Reported as its own line, because
        distributing it over the named stages would be a guess.
        """
        return (
            self.total_ms - self.escalation_ms - self.retrieval_ms - self.generation_ms
        )

    @property
    def usd(self) -> float:
        """What this turn cost, each model's tokens at that model's price."""
        total = 0.0
        for model, counted in self.tokens_by_model.items():
            if not counted["input"] and not counted["output"]:
                continue
            price = price_for(model)
            total += (
                counted["input"] * price["input"] + counted["output"] * price["output"]
            ) / 1_000_000
        return total


def price_for(model: str) -> dict:
    """The published price of one model, snapshot names included.

    OpenAI answers a request for "gpt-4o-mini" with a reply naming the
    snapshot behind it, "gpt-4o-mini-2024-07-18", and the price table is
    keyed by the alias. The longest alias the name begins with is the model
    that was billed.

    Args:
        model: The model name as the reply gave it.

    Returns:
        Its input and output price per million tokens.

    Raises:
        KeyError: If no priced model matches. Deliberately loud: an unpriced
            model counted as free would understate a whole path silently.
    """
    if model in PRICE_PER_MILLION:
        return PRICE_PER_MILLION[model]
    matches = [name for name in PRICE_PER_MILLION if model.startswith(name)]
    if not matches:
        raise KeyError(
            f"no published price for {model!r}. Add it to PRICE_PER_MILLION "
            f"in scripts/evaluate_system.py; the turns of this run are in "
            f"{TURNS_PATH} and --from-turns will rebuild the report for free."
        )
    return PRICE_PER_MILLION[max(matches, key=len)]


def record_turn(turn: Turn, mode: str) -> None:
    """Append one finished turn to the file that survives a later failure.

    Args:
        turn: The turn just measured.
        mode: "sequential" or "burst", so the two passes stay apart.
    """
    TURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(turn)
    record["mode"] = mode
    with open(TURNS_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + chr(10))


def read_turns(path: Path, mode: str) -> list[Turn]:
    """Read back the turns of the last run in this file.

    Args:
        path: The JSONL the run wrote.
        mode: Which pass to return, "sequential" or "burst".

    Returns:
        The turns of that pass, from the most recent run in the file.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.pop("mode") == mode:
                records.append(record)
    turns, seen = [], set()
    # A file can hold several runs; the last one wins, and a run is the tail
    # of the file after the message that starts repeating.
    for record in reversed(records):
        if record["text"] in seen:
            break
        seen.add(record["text"])
        turns.append(Turn(**record))
    return list(reversed(turns))


def load_messages(path: Path, per_path: int) -> list[tuple[str, str, str]]:
    """Choose the messages to send, evenly across the languages.

    The set is the one group B was scored on, so the messages are real
    examples of each path and were read by hand. They are taken in file
    order, round-robin by language, so a re-run sends the same messages.

    Args:
        path: The labelled test set.
        per_path: How many messages to take for each path.

    Returns:
        Triples of text, path and language.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    chosen = []
    for wanted in PATHS:
        by_language = defaultdict(list)
        for row in rows:
            if row["intent"] == wanted:
                by_language[row.get("language", "")].append(row["text"])
        languages = sorted(by_language)
        taken, position = [], 0
        while len(taken) < per_path and any(by_language.values()):
            bucket = by_language[languages[position % len(languages)]]
            if bucket:
                taken.append(bucket.pop(0))
            position += 1
        for text in taken[:per_path]:
            language = next(r.get("language", "") for r in rows if r["text"] == text)
            chosen.append((text, wanted, language))
    return chosen


async def send(orchestrator, classifier, text, expected_path, language) -> Turn:
    """Send one message through the orchestrator and time every stage.

    Args:
        orchestrator: The compiled graph the bot runs.
        classifier: The local intent classifier.
        text: The message to send.
        expected_path: The path this message was chosen to exercise.
        language: Its language, for the per-language breakdown.

    Returns:
        One measured turn.
    """
    # Free and deterministic, so running it here changes no measurement and
    # tells the breakdown whether the first model call was an escalation.
    started = time.perf_counter()
    _, confidence = classify(classifier, text)
    local_ms = (time.perf_counter() - started) * 1000
    escalated = confidence < CONFIDENCE_THRESHOLD

    # The handler is bound to the graph rather than passed at the call, so
    # the bot's own process_message() is still what runs the message.
    # LangChain merges a bound config with the one process_message builds,
    # and LangGraph carries callbacks down into every node.
    spans = Spans()
    watched = orchestrator.with_config(callbacks=[spans])
    thread = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    started = time.perf_counter()
    result = await process_message(watched, text, thread_id=f"latency-{thread}")
    total_ms = (time.perf_counter() - started) * 1000

    calls = list(spans.model_ms)
    escalation_ms = calls.pop(0) if escalated and calls else 0.0
    generation_ms = calls[-1] if calls else 0.0
    return Turn(
        text=text,
        expected_path=expected_path,
        intent=result.intent,
        language=language,
        total_ms=total_ms,
        local_ms=local_ms,
        escalated=escalated,
        escalation_ms=escalation_ms,
        retrieval_ms=spans.retrieval_ms,
        generation_ms=generation_ms,
        input_tokens=spans.input_tokens,
        output_tokens=spans.output_tokens,
        tokens_by_model=spans.tokens_by_model,
    )


def percentile(values: list[float], share: float) -> float:
    """The value below which that share of the sample falls."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(share * len(ordered))) - 1))
    return ordered[index]


def summarise_path(turns: list[Turn]) -> dict:
    """Reduce one path's turns to the figures A1, A2 and A4 report."""
    totals = [t.total_ms for t in turns]
    return {
        "n": len(turns),
        "latency": {
            "mean_ms": statistics.fmean(totals),
            "median_ms": percentile(totals, 0.5),
            "p95_ms": percentile(totals, 0.95),
            "min_ms": min(totals),
            "max_ms": max(totals),
        },
        "stages": {
            "escalation_ms": statistics.fmean([t.escalation_ms for t in turns]),
            "retrieval_ms": statistics.fmean([t.retrieval_ms for t in turns]),
            "generation_ms": statistics.fmean([t.generation_ms for t in turns]),
            "other_ms": statistics.fmean([t.other_ms for t in turns]),
            "local_classifier_ms": statistics.fmean([t.local_ms for t in turns]),
            "note": (
                "the three model and retriever spans are measured by "
                "callbacks; other_ms is what they leave over, and holds the "
                "local classifier and LangGraph's own dispatch. "
                "local_classifier_ms times the same deterministic call "
                "outside the graph, as a proxy for its share of other_ms."
            ),
        },
        "escalated": sum(1 for t in turns if t.escalated),
        "labelled_as_another_path": [
            {"text": t.text, "labelled": t.expected_path}
            for t in turns
            if t.intent != t.expected_path
        ],
        "cost": {
            "models": sorted(
                {model for t in turns for model in t.tokens_by_model if model}
            ),
            "input_tokens_mean": statistics.fmean([t.input_tokens for t in turns]),
            "output_tokens_mean": statistics.fmean([t.output_tokens for t in turns]),
            "usd_per_message": statistics.fmean([t.usd for t in turns]),
            "usd_per_10_message_conversation": statistics.fmean([t.usd for t in turns])
            * 10,
        },
    }


async def run_sequentially(orchestrator, classifier, messages) -> list[Turn]:
    """Send every message one at a time, which is what A1 and A2 measure."""
    turns = []
    for position, (text, path, language) in enumerate(messages, start=1):
        turn = await send(orchestrator, classifier, text, path, language)
        record_turn(turn, "sequential")
        turns.append(turn)
        if position % 10 == 0:
            print(f"  {position}/{len(messages)} sent")
    return turns


async def run_in_bursts(orchestrator, classifier, messages) -> list[Turn]:
    """Send the messages five at a time, which is what A3 measures."""
    turns = []
    for start in range(0, len(messages), BURST_SIZE):
        batch = messages[start : start + BURST_SIZE]
        measured = await asyncio.gather(
            *(
                send(orchestrator, classifier, text, path, language)
                for text, path, language in batch
            )
        )
        for turn in measured:
            record_turn(turn, "burst")
        turns.extend(measured)
        print(f"  {len(turns)}/{len(messages)} sent, {len(batch)} at a time")
    return turns


def build_the_bot() -> object:
    """Build the orchestrator the way the bot's post_init hook builds it.

    No checkpointer: every message is a first message, which is the case
    these figures are about.

    Returns:
        The compiled graph.
    """
    return build_orchestrator(
        build_rag_chain(get_retriever(load_vectorstore())),
        build_translation_chain(),
        None,
    )


def print_report(report: dict) -> None:
    """Print the figures the journal entry records."""
    print("\n" + "=" * 70)
    print(f"end-to-end latency on {report['machine']['label']}")
    print("=" * 70)
    for path, block in report["paths"].items():
        latency = block["latency"]
        stages = block["stages"]
        print(f"\n[{path}] n={block['n']}")
        print(
            f"  total    mean {latency['mean_ms']:>8.1f} ms"
            f"   median {latency['median_ms']:>8.1f}"
            f"   p95 {latency['p95_ms']:>8.1f}"
        )
        print(
            f"  stages   escalation {stages['escalation_ms']:>7.1f}"
            f"   retrieval {stages['retrieval_ms']:>7.1f}"
            f"   generation {stages['generation_ms']:>7.1f}"
            f"   other {stages['other_ms']:>6.1f}"
            f"   (local classifier {stages['local_classifier_ms']:.2f})"
        )
        print(f"  escalated to the model: {block['escalated']} of {block['n']}")
        if block["n"] < DEFAULT_PER_PATH:
            print(
                f"  FEWER THAN {DEFAULT_PER_PATH}: section 4.5 asks for"
                f" {DEFAULT_PER_PATH} per path, this has {block['n']}"
            )
        print(f"  models {', '.join(block['cost']['models']) or 'none'}")
        print(
            f"  cost ${block['cost']['usd_per_message']:.6f} per message,"
            f" ${block['cost']['usd_per_10_message_conversation']:.5f}"
            f" per 10-message conversation"
        )
        for stray in block["labelled_as_another_path"]:
            print(
                f"  arrived here labelled {stray['labelled']}:" f" {stray['text'][:56]}"
            )

    if "burst" in report:
        print("\n[A3] five messages at a time, against one at a time")
        print(f"  {'':<16}{'mean':>10}{'p95':>10}")
        for name, block in (
            ("one at a time", report["sequential_overall"]),
            (f"{BURST_SIZE} at a time", report["burst"]),
        ):
            print(f"  {name:<16}{block['mean_ms']:>10.1f}{block['p95_ms']:>10.1f}")
    print(
        f"\n[cost] ${report['cost_of_this_run_usd']:.4f} for"
        f" {report['messages_sent']} messages"
    )


def main() -> None:
    """Measure the paths and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-path",
        type=int,
        default=DEFAULT_PER_PATH,
        help=f"messages per path (default {DEFAULT_PER_PATH}, section 4.5 asks 30)",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help=f"also send the same messages {BURST_SIZE} at a time, for A3",
    )
    parser.add_argument(
        "--label",
        default="",
        help="what machine this is, e.g. 'raspberry pi 4, 4gb'",
    )
    parser.add_argument(
        "--from-turns",
        action="store_true",
        help=f"rebuild the report from {TURNS_PATH}, sending nothing",
    )
    arguments = parser.parse_args()

    if arguments.from_turns:
        turns = read_turns(TURNS_PATH, "sequential")
        print(f"{len(turns)} turns read from {TURNS_PATH}, nothing sent")
    else:
        messages = load_messages(TEST_SET_PATH, arguments.per_path)
        print(f"{len(messages)} messages, {arguments.per_path} per path")

        classifier = load_classifier()
        orchestrator = build_the_bot()
        turns = asyncio.run(run_sequentially(orchestrator, classifier, messages))
    # Grouped by the path the message actually took, not by the label it
    # came with: the wait belongs to the work that was done. A message the
    # classifier routed elsewhere is listed under the path it reached, and
    # named there as having carried another label.
    by_path = defaultdict(list)
    for turn in turns:
        by_path[turn.intent].append(turn)

    totals = [t.total_ms for t in turns]
    report = {
        "machine": {
            "label": arguments.label or platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.machine(),
            "python": platform.python_version(),
        },
        "settings": {
            "model": LLM_MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "checkpointer": None,
            "note": (
                "no conversation behind any message, so the knowledge chain "
                "does not buy the extra call that rewrites a follow-up"
            ),
        },
        "messages_sent": len(turns),
        "paths": {path: summarise_path(rows) for path, rows in by_path.items()},
        "sequential_overall": {
            "mean_ms": statistics.fmean(totals),
            "p95_ms": percentile(totals, 0.95),
        },
        "cost_of_this_run_usd": sum(t.usd for t in turns),
    }

    if arguments.from_turns:
        burst = read_turns(TURNS_PATH, "burst")
    elif arguments.concurrent:
        print(f"\nnow {BURST_SIZE} at a time")
        burst = asyncio.run(run_in_bursts(orchestrator, classifier, messages))
    else:
        burst = []

    if burst:
        burst_totals = [t.total_ms for t in burst]
        report["burst"] = {
            "size": BURST_SIZE,
            "n": len(burst),
            "mean_ms": statistics.fmean(burst_totals),
            "median_ms": percentile(burst_totals, 0.5),
            "p95_ms": percentile(burst_totals, 0.95),
        }
        report["cost_of_this_run_usd"] += sum(t.usd for t in burst)
        report["messages_sent"] += len(burst)

    print_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
