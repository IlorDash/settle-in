"""Count the verdicts of the manual grounding pass - measurement C3.

The pass itself is done by hand in data/eval_retrieval_review.md, which
scripts/evaluate_retrieval.py writes. This script only reads the marks back:
it refuses a sheet where a section is unmarked or marked twice, counts the
verdicts, and compares the `declined` marks with the declines the automatic
detector found, so a disagreement between the two is visible rather than
buried.

Run from the repo root, free, no API call:

    python -m scripts.tally_grounding
"""

import json
import re
from pathlib import Path

REVIEW_PATH = Path("data/eval_retrieval_review.md")
REPORT_PATH = Path("data/eval_retrieval.json")

VERDICTS = ("grounded", "ungrounded", "declined")
SECTION = re.compile(r"^## (\d+)\. \[(\w+)\] (.+)$", re.M)


def read_sheet(path: Path) -> dict[int, dict]:
    """Read every section of the review sheet.

    Args:
        path: The sheet the pass was made on.

    Returns:
        One entry per query: the marks found in its verdict line and the
        text of its notes.
    """
    text = path.read_text(encoding="utf-8")
    starts = [(int(m.group(1)), m.start()) for m in SECTION.finditer(text)]
    sections = {}
    for position, (query_id, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(text)
        block = text[start:end]
        marked = [
            verdict
            for verdict in VERDICTS
            if re.search(rf"\[[xX]\]\s*{verdict}\b", block)
        ]
        # Everything after the marker up to the next heading or rule, so an
        # empty Notes line does not swallow the section that follows it.
        notes = re.search(r"\*\*Notes:\*\*(.*?)(?=\n#{2,3} |\n---|\Z)", block, re.S)
        sections[query_id] = {
            "marked": marked,
            "notes": notes.group(1).strip() if notes else "",
        }
    return sections


def main() -> None:
    """Count the marks and report what does not add up."""
    if not REVIEW_PATH.exists():
        raise SystemExit(
            f"{REVIEW_PATH} does not exist. Run scripts.evaluate_retrieval first."
        )
    sections = read_sheet(REVIEW_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in report["per_query"]}

    unmarked = sorted(i for i, s in sections.items() if not s["marked"])
    doubled = sorted(i for i, s in sections.items() if len(s["marked"]) > 1)
    print(f"{len(sections)} sections in {REVIEW_PATH}")
    if unmarked:
        print(f"  UNMARKED: {unmarked}")
    if doubled:
        print(f"  MARKED TWICE: {doubled}")

    counts = {verdict: 0 for verdict in VERDICTS}
    answerable = {verdict: 0 for verdict in VERDICTS}
    for query_id, section in sections.items():
        if len(section["marked"]) != 1:
            continue
        verdict = section["marked"][0]
        counts[verdict] += 1
        if by_id.get(query_id, {}).get("expected") != "none":
            answerable[verdict] += 1

    decided = sum(counts.values())
    print(f"\n{decided} of {len(sections)} sections decided")
    for verdict in VERDICTS:
        print(
            f"  {verdict:<12} {counts[verdict]:>3}"
            f"   (answerable only: {answerable[verdict]})"
        )
    judged = answerable["grounded"] + answerable["ungrounded"]
    if judged:
        print(
            f"\nC3 on the answerable queries: {answerable['grounded']} of"
            f" {judged} grounded ({answerable['grounded'] / judged:.4f})"
        )

    # Where the hand pass and the automatic detector disagree, the hand pass
    # is the measurement and the disagreement is worth printing.
    for query_id, section in sections.items():
        if len(section["marked"]) != 1:
            continue
        by_hand = section["marked"][0] == "declined"
        automatic = by_id.get(query_id, {}).get("declined")
        if automatic is not None and by_hand != automatic:
            reading = "declined" if automatic else "answered"
            print(
                f"  DISAGREEMENT on {query_id}: the sheet says"
                f" {section['marked'][0]}, the detector read it as {reading}"
            )

    notes = {i: s["notes"] for i, s in sections.items() if s["notes"]}
    if notes:
        print(f"\n{len(notes)} section(s) carry notes:")
        for query_id, note in sorted(notes.items()):
            print(f"  {query_id}: {note.splitlines()[0]}")

    if unmarked or doubled:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
