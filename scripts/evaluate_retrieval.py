"""Score the knowledge agent on the labelled queries - measurement group C.

Section 4.2 rests the knowledge agent on two decisions, and this script
checks both of them on the same set of queries:

  * C1, the hit rate. Thirty queries are each written so that one known
    document holds the answer. Whether that document appears among the four
    passages the retriever returns is the first decision.
  * C2, the decline rate. The agent is supposed to decline when the passages
    do not carry an answer. Declining is counted apart from accuracy, and it
    is read twice: on the thirty answerable queries a decline is a failure,
    and on the ten probes whose answer is nowhere in the corpus a decline is
    the designed behaviour.

C3 - whether a sentence of the answer is supported by a sentence of the
passages - is a manual pass. No automatic check settles it, so this script
writes every answer next to the passages the prompt actually carried, into
data/eval_retrieval_review.md, and the verdicts are filled in there by hand.

Three things this script has to get right:

  * It runs the chain the bot runs. `build_rag_chain` and `get_retriever` are
    the bot's own, so the prompt, the model and k are whatever the deployed
    system uses; re-implementing retrieval here would measure a pipeline
    nobody ships.
  * The retrieval it scores is the retrieval that fed the answer. Both come
    from the same pass over the query, so a hit rate and a grounding verdict
    can never be about different passages.
  * The answer cache is keyed by the query and by what was retrieved for it.
    Re-running is then free while the index stands, and rebuilding the index
    re-buys the answers instead of silently reporting old ones.

Run from the repo root:

    python -m scripts.evaluate_retrieval --retrieval-only   # C1 only, ~free
    python -m scripts.evaluate_retrieval --limit 5          # a fraction of a cent
    python -m scripts.evaluate_retrieval                    # C1, C2, review sheet
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from scripts.evaluate_system import PRICE_PER_MILLION, UsageRecorder
from src.agents.rag_agent import LLM_MODEL, build_rag_chain
from src.knowledge.vectorstore import (
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    get_retriever,
    load_vectorstore,
)

TEST_SET_PATH = Path("data/retrieval_test.csv")
CACHE_PATH = Path("data/eval_cache_rag.jsonl")
REPORT_PATH = Path("data/eval_retrieval.json")
REVIEW_PATH = Path("data/eval_retrieval_review.md")

# The label the test set gives a query whose answer is in no document.
KIND_NONE = "none"

# What the system prompt tells the agent to say when the passages carry no
# answer. The model is asked for this sentence in English, but it answers a
# Russian or Serbian question in that language and translates the refusal
# with it, so the shorter markers below are needed as well. Every decline is
# written into the review sheet, and the manual pass is what settles the
# borderline ones.
DECLINE_SENTENCE = "i don't have enough information to answer that question"
#
# The Russian list was extended after the first paid run: the model wrote
# "Я не имею достаточно информации", which none of the original markers
# caught, and the query was reported as answered when it had been declined.
# It was found by reading the review sheet, which is the argument for
# writing every answer out rather than trusting the rule.
DECLINE_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "достаточно информации",
    "достаточной информации",
    "достаточно данных",
    "не хватает информации",
    "nemam dovoljno informacija",
    "nema dovoljno informacija",
    "nemam dovoljno podataka",
    "nedovoljno informacija",
)


@dataclass(frozen=True)
class Query:
    """One labelled query.

    Attributes:
        id: Its row number in the test set, used to name it in reports.
        language: The language it is written in.
        kind: "plain", "overlap" for a query whose topic another document
            also touches, or "none" for one the corpus cannot answer.
        document: The file that holds the answer, or "none".
        text: The query as a user would send it.
        expected_fact: What a correct answer has to contain. Read by the
            person doing the C3 pass; nothing automatic uses it.
    """

    id: int
    language: str
    kind: str
    document: str
    text: str
    expected_fact: str

    @property
    def answerable(self) -> bool:
        """Whether some document in the corpus holds the answer."""
        return self.kind != KIND_NONE


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, as it reached the prompt.

    Attributes:
        source: The document it came from, as a file name.
        start_index: Where the chunk starts in that document, so a reader
            of the review sheet can find it.
        text: The chunk itself.
    """

    source: str
    start_index: int
    text: str


@dataclass(frozen=True)
class Result:
    """What one query produced.

    Attributes:
        query: The query it answers.
        passages: The chunks the retriever returned, in its own order.
        answer: What the agent replied, or None under --retrieval-only.
        retrieval_ms: Wall-clock time to embed the query and search.
        generation_ms: Wall-clock time for the answer, or 0.0.
        input_tokens: Prompt tokens billed for the answer, or 0 when the
            answer came from the cache.
        output_tokens: Completion tokens billed, or 0.
        cached: Whether the answer was read back rather than bought.
    """

    query: Query
    passages: list[Passage]
    answer: str | None
    retrieval_ms: float
    generation_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    @property
    def sources(self) -> list[str]:
        """The documents behind the passages, in retrieval order."""
        return [passage.source for passage in self.passages]

    @property
    def hit(self) -> bool:
        """Whether the document holding the answer was retrieved at all."""
        return self.query.answerable and self.query.document in self.sources

    @property
    def rank(self) -> int | None:
        """Position of the first passage from the right document, 1-based."""
        if not self.hit:
            return None
        return self.sources.index(self.query.document) + 1

    @property
    def declined(self) -> bool:
        """Whether the answer is a refusal to answer."""
        return self.answer is not None and looks_like_a_decline(self.answer)


@dataclass
class AnswerCache:
    """Answers bought so far, keyed by the query and its retrieved context.

    Attributes:
        path: The JSONL file it is written to.
        entries: What has been read back from that file.

    The context is part of the key on purpose. An answer is a function of the
    question and of the passages that reached the prompt, so a rebuilt index
    that retrieves different passages has to be paid for again; keying on the
    query alone would report an old answer as though it belonged to the new
    retrieval.
    """

    path: Path
    entries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path) -> "AnswerCache":
        """Load whatever the file already holds."""
        cache = cls(path=path)
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    cache.entries[record["key"]] = record
        return cache

    def get(self, key: str) -> dict | None:
        """Return the record for a query and its context, or None."""
        return self.entries.get(key)

    def put(self, record: dict) -> None:
        """Store a record and flush it to disk immediately."""
        self.entries[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cache_key(query: Query, passages: list[Passage]) -> str:
    """Name one (question, retrieved context) pair.

    Args:
        query: The query that was asked.
        passages: What the retriever returned for it.

    Returns:
        A short hash of the query text, the model and the passages.
    """
    material = "\n".join(
        [query.text, LLM_MODEL]
        + [f"{p.source}:{p.start_index}:{len(p.text)}" for p in passages]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def basename(source: str) -> str:
    r"""Reduce a chunk's stored source path to a file name, on any platform.

    Args:
        source: The `source` metadata Chroma stored when the index was built.

    Returns:
        The file name at the end of it.

    Chroma keeps this string exactly as the machine that built the index
    wrote it, so an index built on Windows carries a source such as
    `data\knowledge_base\01_white_card_registration.txt` into whatever
    reads it later. `pathlib.Path` resolves against the platform it runs
    on, and a PosixPath does not treat a backslash as a separator: on the
    Raspberry Pi every source came back whole, matched no expected
    document, and C1 scored 0.0000 on retrieval that was in fact perfect.
    Normalising here keeps the score a property of the index rather than
    of the machine reading it. The bot is unaffected either way, since
    `src/agents/rag_agent.py` never reads this metadata, so the defect
    lived wholly in the instrument.
    """
    return PurePosixPath(str(source).replace("\\", "/")).name


def looks_like_a_decline(answer: str) -> bool:
    """Say whether an answer is the agent declining to answer.

    Args:
        answer: What the agent replied.

    Returns:
        True if the reply carries the refusal the system prompt asks for, in
        any of the three languages the bot is used in.
    """
    lowered = answer.lower().replace("’", "'")
    return any(marker in lowered for marker in DECLINE_MARKERS)


def load_queries(path: Path) -> list[Query]:
    """Read the labelled queries.

    Args:
        path: The CSV holding them.

    Returns:
        Every row of it, in file order.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        return [
            Query(
                id=int(row["id"]),
                language=row["language"],
                kind=row["kind"],
                document=row["document"],
                text=row["query"],
                expected_fact=row["expected_fact"],
            )
            for row in csv.DictReader(handle)
        ]


def retrieve(retriever, query: Query) -> tuple[list[Passage], float]:
    """Search the store for one query and time the search.

    Args:
        retriever: The retriever the bot uses.
        query: The query to search on.

    Returns:
        The passages it returned, and the wall-clock milliseconds it took.
    """
    started = time.perf_counter()
    documents = retriever.invoke(query.text)
    elapsed = (time.perf_counter() - started) * 1000
    passages = [
        Passage(
            source=basename(document.metadata.get("source", "?")),
            start_index=int(document.metadata.get("start_index", -1)),
            text=document.page_content,
        )
        for document in documents
    ]
    return passages, elapsed


async def answer(chain, query: Query) -> tuple[str, float, int, int]:
    """Send one query through the chain the bot runs.

    Args:
        chain: The chain built by build_rag_chain().
        query: The query to answer.

    Returns:
        The answer, the wall-clock milliseconds, and the tokens billed.
    """
    recorder = UsageRecorder()
    started = time.perf_counter()
    reply = await chain.ainvoke(
        {"input": query.text, "history": []}, config={"callbacks": [recorder]}
    )
    elapsed = (time.perf_counter() - started) * 1000
    return reply, elapsed, recorder.input_tokens, recorder.output_tokens


async def run(
    queries: list[Query], retrieval_only: bool, refresh: bool
) -> list[Result]:
    """Retrieve for every query, and answer it unless told not to.

    Args:
        queries: The labelled queries.
        retrieval_only: Skip the answers, which is what makes a run free.
        refresh: Ignore the cache and buy every answer again.

    Returns:
        One result per query, in the order the queries were given.
    """
    retriever = get_retriever(load_vectorstore())
    chain = None if retrieval_only else build_rag_chain(retriever)
    cache = AnswerCache.open(CACHE_PATH)
    results, bought = [], 0

    for position, query in enumerate(queries, start=1):
        passages, retrieval_ms = retrieve(retriever, query)
        if retrieval_only:
            results.append(Result(query, passages, None, retrieval_ms))
            continue

        key = cache_key(query, passages)
        record = None if refresh else cache.get(key)
        if record is None:
            reply, generation_ms, input_tokens, output_tokens = await answer(
                chain, query
            )
            record = {
                "key": key,
                "id": query.id,
                "query": query.text,
                "answer": reply,
                "generation_ms": generation_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "sources": [p.source for p in passages],
            }
            cache.put(record)
            bought += 1
            results.append(
                Result(
                    query,
                    passages,
                    reply,
                    retrieval_ms,
                    generation_ms,
                    input_tokens,
                    output_tokens,
                )
            )
        else:
            results.append(
                Result(
                    query,
                    passages,
                    record["answer"],
                    retrieval_ms,
                    record["generation_ms"],
                    cached=True,
                )
            )
        if position % 10 == 0:
            print(f"  {position}/{len(queries)} done, {bought} bought")

    if retrieval_only:
        print(f"  {len(queries)} retrieved, no answers generated")
    else:
        print(f"  {len(queries)} done, {bought} bought, {len(queries) - bought} cached")
    return results


def percentile(values: list[float], share: float) -> float:
    """The value below which that share of the sample falls."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(share * len(ordered))) - 1))
    return ordered[index]


def share(part: int, whole: int) -> float:
    """A proportion, or 0.0 when there is nothing to divide."""
    return part / whole if whole else 0.0


def summarise(results: list[Result], retrieval_only: bool) -> dict:
    """Turn the results into the figures C1 and C2 report.

    Args:
        results: One result per query.
        retrieval_only: Whether answers were produced at all.

    Returns:
        A report, saved as data/eval_retrieval.json.
    """
    answerable = [r for r in results if r.query.answerable]
    unanswerable = [r for r in results if not r.query.answerable]
    hits = [r for r in answerable if r.hit]
    ranks = [r.rank for r in hits]

    report = {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "settings": {
            "embedding_model": EMBEDDING_MODEL,
            "k": RETRIEVER_TOP_K,
            "llm_model": LLM_MODEL,
        },
        "c1": {
            "hit_rate": share(len(hits), len(answerable)),
            "hits": len(hits),
            "hit_at_1": share(sum(1 for r in hits if r.rank == 1), len(answerable)),
            "mean_rank_of_hit": statistics.fmean(ranks) if ranks else None,
            "misses": [
                {
                    "id": r.query.id,
                    "query": r.query.text,
                    "expected": r.query.document,
                    "retrieved": r.sources,
                }
                for r in answerable
                if not r.hit
            ],
            "by_language": {},
            "by_kind": {},
            "by_document": {},
        },
        "per_query": [
            {
                "id": r.query.id,
                "language": r.query.language,
                "kind": r.query.kind,
                "expected": r.query.document,
                "retrieved": r.sources,
                "rank": r.rank,
                "declined": r.declined if r.answer is not None else None,
            }
            for r in results
        ],
        "retrieval_latency": {
            "mean_ms": statistics.fmean([r.retrieval_ms for r in results]),
            "p95_ms": percentile([r.retrieval_ms for r in results], 0.95),
            "note": (
                "embedding the query over the network plus the search in "
                "Chroma. The round trip dominates, so this is a figure about "
                "the connection as much as the store, and the first query of "
                "a run carries the connection setup."
            ),
        },
    }

    for name, key in (("by_language", "language"), ("by_kind", "kind")):
        for value in sorted({getattr(r.query, key) for r in answerable}):
            group = [r for r in answerable if getattr(r.query, key) == value]
            report["c1"][name][value] = {
                "n": len(group),
                "hit_rate": share(sum(1 for r in group if r.hit), len(group)),
            }
    for document in sorted({r.query.document for r in answerable}):
        group = [r for r in answerable if r.query.document == document]
        report["c1"]["by_document"][document] = {
            "n": len(group),
            "hits": sum(1 for r in group if r.hit),
        }

    if retrieval_only:
        return report

    wrong_declines = [r for r in answerable if r.declined]
    right_declines = [r for r in unanswerable if r.declined]
    report["c2"] = {
        "declines_on_answerable": len(wrong_declines),
        "decline_rate_on_answerable": share(len(wrong_declines), len(answerable)),
        "declines_on_unanswerable": len(right_declines),
        "decline_rate_on_unanswerable": share(len(right_declines), len(unanswerable)),
        "declined_although_the_document_was_retrieved": sum(
            1 for r in wrong_declines if r.hit
        ),
        # A decline on an answerable query is only the agent's fault if the
        # passage carrying the answer was in front of it. The chunk offsets
        # are printed so that can be checked: the right document reaching the
        # prompt does not mean the right chunk of it did.
        "declined_on_answerable": [
            {
                "id": r.query.id,
                "query": r.query.text,
                "expected_fact": r.query.expected_fact,
                "retrieved": [f"{p.source}@{p.start_index}" for p in r.passages],
            }
            for r in wrong_declines
        ],
        "answered_although_nothing_was_stored": [
            {"id": r.query.id, "query": r.query.text, "answer": r.answer}
            for r in unanswerable
            if not r.declined
        ],
        "declines_not_in_the_english_wording": [
            r.query.id
            for r in results
            if r.declined and DECLINE_SENTENCE not in r.answer.lower()
        ],
    }

    generation = [r.generation_ms for r in results if r.generation_ms]
    report["generation_latency"] = {
        "mean_ms": statistics.fmean(generation) if generation else None,
        "p95_ms": percentile(generation, 0.95) if generation else None,
        "note": (
            "the whole chain, question in to answer out, wall clock on the "
            "machine named in the journal entry. The chain does its own "
            "retrieval, so this figure contains a search of its own and is "
            "not additive with retrieval_latency, which is a separate call "
            "made to score C1."
        ),
    }
    price = PRICE_PER_MILLION[LLM_MODEL]
    bought = [r for r in results if not r.cached]
    input_tokens = sum(r.input_tokens for r in bought)
    output_tokens = sum(r.output_tokens for r in bought)
    report["cost"] = {
        "answers_bought_this_run": len(bought),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd_for_this_run": (
            input_tokens * price["input"] + output_tokens * price["output"]
        )
        / 1_000_000,
        "note": (
            "the answers only. Embedding each query is billed separately at "
            "the embedding model's rate and is well under a cent for this set, "
            "and the retriever reports no token count for it."
        ),
    }
    return report


def write_review_sheet(results: list[Result], path: Path) -> None:
    """Write the sheet the C3 grounding pass is done on.

    Each query gets the passages that reached the prompt and the answer that
    came back, so the reader can check a sentence of the answer against a
    sentence of the passages without running anything.

    Args:
        results: One result per query.
        path: Where to write the sheet.
    """
    lines = [
        "# Group C, manual grounding pass (C3)",
        "",
        "One section per query, holding the answer the agent gave and the four",
        "passages that were in the prompt when it gave it. Nothing here is",
        "filled in automatically: the marks below are what the C3 entry in",
        "thesis/measurements.md counts.",
        "",
        "**How to mark one section.** Read the answer first, then the passages.",
        "Put an `x` in exactly one box:",
        "",
        "- `grounded` - every factual claim in the answer is supported by a",
        "  sentence of the passages printed under it. Numbers, dates, fees and",
        "  names have to match the passages, not your own knowledge of Serbia.",
        "- `ungrounded` - at least one claim is not in the passages, or",
        "  contradicts them. A figure that is right in the world but absent",
        "  from the passages is ungrounded: it was not retrieved, so the agent",
        "  did not read it there.",
        "- `declined` - the agent refused to answer. Nothing to judge for",
        "  grounding; whether the refusal was correct is C2's question, not",
        "  this one.",
        "",
        "Add a line under **Notes** when the verdict needed a decision, so the",
        "entry can say why. Then run:",
        "",
        "```",
        ".venv/Scripts/python.exe -m scripts.tally_grounding",
        "```",
        "",
        "which counts the marks, refuses a section that is unmarked or marked",
        "twice, and compares your `declined` marks with the ones the automatic",
        "detector found.",
        "",
        "---",
        "",
    ]
    for result in results:
        query = result.query
        expected = query.document if query.answerable else "nothing - not in the corpus"
        lines += [
            f"## {query.id}. [{query.language}] {query.text}",
            "",
            f"- **Answer should come from:** {expected}",
            f"- **Expected fact:** {query.expected_fact}",
            f"- **Retrieved:** {', '.join(result.sources)}",
            f"- **Document retrieved:** {'yes' if result.hit else 'no'}"
            + (f" at position {result.rank}" if result.hit else ""),
            f"- **Reads as a decline:** {'yes' if result.declined else 'no'}",
            "",
            "**Verdict:** [ ] grounded  [ ] ungrounded  [ ] declined",
            "",
            "**Notes:**",
            "",
            "### The answer",
            "",
            (result.answer or "_not generated - run without --retrieval-only_"),
            "",
            "### The passages the prompt carried",
            "",
        ]
        for position, passage in enumerate(result.passages, start=1):
            lines += [
                f"**{position}. {passage.source}** (from character "
                f"{passage.start_index})",
                "",
                "```",
                passage.text.strip(),
                "```",
                "",
            ]
        lines += ["---", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    # A filled-in sheet is hand work that no re-run may destroy. If any box
    # has been ticked, the new sheet goes beside it and the reader decides.
    if path.exists() and re.search(r"\[[xX]\]", path.read_text(encoding="utf-8")):
        beside = path.with_suffix(".new.md")
        beside.write_text("\n".join(lines), encoding="utf-8")
        print(
            f"\n{path} already carries verdicts, so it was left alone.\n"
            f"The sheet for this run is {beside}."
        )
        return
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: dict) -> None:
    """Print the figures the journal entry records."""
    print("\n" + "=" * 70)
    print(f"retrieval evaluation, {report['n']} queries")
    print("=" * 70)
    c1 = report["c1"]
    print(
        f"\n[C1] hit rate {c1['hit_rate']:.4f}"
        f" ({c1['hits']} of {report['n_answerable']}),"
        f" hit@1 {c1['hit_at_1']:.4f}"
    )
    if c1["mean_rank_of_hit"] is not None:
        print(
            f"  mean rank of the right document when found:"
            f" {c1['mean_rank_of_hit']:.2f} of {report['settings']['k']}"
        )
    for name in ("by_language", "by_kind"):
        readable = ", ".join(
            f"{value} {block['hit_rate']:.4f} (n={block['n']})"
            for value, block in c1[name].items()
        )
        print(f"  {name.replace('_', ' ')}: {readable}")
    for miss in c1["misses"]:
        print(
            f"  MISS {miss['id']}: expected {miss['expected']},"
            f" got {miss['retrieved']}"
        )
    print(f"\n[retrieval latency] {report['retrieval_latency']}")

    if "c2" not in report:
        print("\n[C2] not measured - this was a retrieval-only run")
        return
    c2 = report["c2"]
    print(
        f"\n[C2] declined {c2['declines_on_answerable']} of"
        f" {report['n_answerable']} answerable"
        f" ({c2['decline_rate_on_answerable']:.4f}) - each one a failure"
    )
    print(
        f"     declined {c2['declines_on_unanswerable']} of"
        f" {report['n_unanswerable']} unanswerable"
        f" ({c2['decline_rate_on_unanswerable']:.4f}) - each one correct"
    )
    for row in c2["declined_on_answerable"]:
        print(f"  DECLINED {row['id']}: {row['query']}")
        print(f"    it was given {row['retrieved']}")
    if c2["answered_although_nothing_was_stored"]:
        for row in c2["answered_although_nothing_was_stored"]:
            print(f"  ANSWERED ANYWAY {row['id']}: {row['query']}")
    if c2["declines_not_in_the_english_wording"]:
        print(
            f"  declines phrased in another language, confirm by hand:"
            f" {c2['declines_not_in_the_english_wording']}"
        )
    print(f"\n[generation latency] {report['generation_latency']}")
    print(f"[cost] {report['cost']}")


def main() -> None:
    """Run the evaluation and save the report and the review sheet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="use only the first N queries")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score C1 without generating answers, which costs almost nothing",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="buy every answer again instead of reading the cache",
    )
    arguments = parser.parse_args()

    queries = load_queries(TEST_SET_PATH)
    if arguments.limit:
        queries = queries[: arguments.limit]
    print(f"{len(queries)} queries from {TEST_SET_PATH}")

    results = asyncio.run(run(queries, arguments.retrieval_only, arguments.refresh))
    report = summarise(results, arguments.retrieval_only)
    print_report(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {REPORT_PATH.resolve()}")
    if not arguments.retrieval_only:
        write_review_sheet(results, REVIEW_PATH)
        print(f"The C3 pass is done in {REVIEW_PATH.resolve()}")
        print(f"Answers bought are cached in {CACHE_PATH}")


if __name__ == "__main__":
    main()
