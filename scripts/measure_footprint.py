"""Measure what the bot needs to start: memory and start-up time.

Chapter 4 claims the system is deployable on a Raspberry Pi 4. Group A says
what a user waits for once the bot is running and says nothing about what it
takes to get there, so a reader with a 2 GB or 4 GB board cannot tell from
those figures whether the claim includes their machine. This script answers
that: how much resident memory the process holds once it is ready to answer,
and how long it takes to become ready.

It measures the start-up the bot actually performs. `build_orchestrator` with
the real RAG and translation chains is what `post_init` builds, and the
classifier and the vector store are loaded exactly as the bot loads them, so
the figures belong to the shipped system rather than to a sketch of it.

Nothing is sent to a model. The default run makes no API call at all and is
free; `--retrieve` adds one embedding round trip, so that the peak includes a
real search, and costs a fraction of a cent.

Resident memory is read from `/proc/self/status`, so the figures below are
Linux-only and the peak is the kernel's own high-water mark rather than a
sample this script happened to catch. On a machine without `/proc` the timings
are still measured and the memory columns are reported as unavailable.

Run from the repo root:

    python -m scripts.measure_footprint              # free
    python -m scripts.measure_footprint --retrieve   # adds one embedding call
"""

import argparse
import json
import platform
import time
from pathlib import Path

REPORT_PATH = Path("data/eval_footprint.json")
STATUS_PATH = Path("/proc/self/status")

# The query the --retrieve pass searches on. It is one of the labelled
# queries, so the search it triggers is the search group C measures.
PROBE_QUERY = "how long does a temporary residence permit take?"


def read_memory_kb() -> dict:
    """Read this process's resident memory and its high-water mark.

    Returns:
        `rss_kb`, what is resident now, and `peak_rss_kb`, the largest it has
        been since the process started. Both are None where `/proc` is not
        available, which is every platform but Linux.
    """
    if not STATUS_PATH.exists():
        return {"rss_kb": None, "peak_rss_kb": None}
    fields = {}
    for line in STATUS_PATH.read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        if name in ("VmRSS", "VmHWM"):
            fields[name] = int(rest.split()[0])
    return {"rss_kb": fields.get("VmRSS"), "peak_rss_kb": fields.get("VmHWM")}


def stage(name: str, started: float, stages: list) -> float:
    """Record how long a start-up stage took and what it left resident.

    Args:
        name: The stage that just finished.
        started: When it began, from `time.perf_counter()`.
        stages: The list to append the record to.

    Returns:
        The moment the stage ended, to start the next one from.
    """
    ended = time.perf_counter()
    stages.append({"stage": name, "ms": (ended - started) * 1000, **read_memory_kb()})
    return ended


def measure(retrieve: bool) -> dict:
    """Start the bot up one stage at a time, timing and weighing each.

    The imports are done here rather than at the top of the file on purpose:
    importing LangChain, Chroma and scikit-learn is itself a large part of
    both figures, and a module-level import would have happened before the
    first measurement and gone uncounted.

    Args:
        retrieve: Whether to add one real search, which calls the embedding
            endpoint and so is the only part of this script that costs money.

    Returns:
        The report, saved as data/eval_footprint.json.
    """
    stages = []
    begin = time.perf_counter()
    stages.append({"stage": "interpreter", "ms": 0.0, **read_memory_kb()})

    from src.agents.intent_classifier import load_classifier
    from src.agents.orchestrator import build_orchestrator
    from src.agents.rag_agent import build_rag_chain
    from src.agents.translation_agent import build_translation_chain
    from src.knowledge.vectorstore import get_retriever, load_vectorstore

    started = stage("imports", begin, stages)

    classifier = load_classifier()
    started = stage("classifier", started, stages)

    retriever = get_retriever(load_vectorstore())
    started = stage("vector store", started, stages)

    orchestrator = build_orchestrator(
        build_rag_chain(retriever), build_translation_chain(), None
    )
    started = stage("graph", started, stages)
    ready_ms = (started - begin) * 1000

    # Everything above is local. Only this touches the network.
    if retrieve:
        retriever.invoke(PROBE_QUERY)
        stage("one retrieval", started, stages)

    # The objects are held to the end so that nothing measured above can be
    # collected before the peak is read.
    assert classifier is not None and orchestrator is not None

    return {
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.machine(),
            "python": platform.python_version(),
        },
        "ready_ms": ready_ms,
        "retrieved": retrieve,
        "stages": stages,
        "peak_rss_kb": read_memory_kb()["peak_rss_kb"],
        "note": (
            "resident memory of one process holding the classifier, the "
            "vector store and the compiled graph. The bot adds its Telegram "
            "polling loop and the SQLite checkpointer on top of this."
        ),
    }


def print_report(report: dict) -> None:
    """Print the figures the journal entry records."""
    print("\n" + "=" * 70)
    print(f"start-up footprint on {report['machine']['processor']}")
    print("=" * 70)
    print(f"\n{'stage':<16}{'ms':>10}{'resident':>14}")
    for row in report["stages"]:
        resident = f"{row['rss_kb'] / 1024:.1f} MB" if row["rss_kb"] else "-"
        print(f"{row['stage']:<16}{row['ms']:>10.1f}{resident:>14}")
    print(f"\nready to answer in {report['ready_ms']:.0f} ms")
    if report["peak_rss_kb"]:
        print(f"peak resident memory {report['peak_rss_kb'] / 1024:.1f} MB")
    else:
        print(
            "resident memory was not available: this machine has no "
            "/proc/self/status, so only the timings above were measured."
        )


def main() -> None:
    """Measure the footprint and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="add one embedding round trip, so the peak includes a search",
    )
    arguments = parser.parse_args()

    report = measure(arguments.retrieve)
    print_report(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
