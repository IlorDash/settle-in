"""Persist user feedback on bot answers for later classifier retraining.

Each thumbs up/down becomes one line in a JSONL file. The store is append-only
(one event at a time) and easy to read back as a labeled dataset -- a 👎 on a
misrouted message tells us exactly where the intent classifier is wrong.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_PATH = Path("data/feedback.jsonl")

VERDICT_UP = "up"
VERDICT_DOWN = "down"


@dataclass
class FeedbackRecord:
    """One user verdict on one bot answer.

    Attributes:
        user_message: The message the user originally sent.
        intent: The intent the classifier predicted for that message.
        verdict: VERDICT_UP or VERDICT_DOWN.
        timestamp: ISO-8601 UTC time the feedback was given.
    """

    user_message: str
    intent: str
    verdict: str
    timestamp: str


def record_feedback(
    user_message: str,
    intent: str,
    verdict: str,
    path: Path = FEEDBACK_PATH,
) -> FeedbackRecord:
    """Append one feedback record to the JSONL store and return it.

    Args:
        user_message: The message the user originally sent.
        intent: The intent the classifier predicted for it.
        verdict: VERDICT_UP or VERDICT_DOWN.
        path: Destination JSONL file (defaults to FEEDBACK_PATH).

    Returns:
        The FeedbackRecord that was written.
    """
    record = FeedbackRecord(
        user_message=user_message,
        intent=intent,
        verdict=verdict,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return record
