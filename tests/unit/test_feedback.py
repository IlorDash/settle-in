import json

from src.bot.feedback import (
    VERDICT_DOWN,
    VERDICT_UP,
    FeedbackRecord,
    record_feedback,
)


def test_record_feedback_appends_one_line_per_record(tmp_path):
    path = tmp_path / "feedback.jsonl"

    record_feedback("как дела", "translation", VERDICT_UP, path=path)
    record_feedback("what is a PIB", "knowledge_question", VERDICT_DOWN, path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_record_feedback_writes_the_fields(tmp_path):
    path = tmp_path / "feedback.jsonl"

    record_feedback("как дела", "translation", VERDICT_UP, path=path)

    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert stored["user_message"] == "как дела"
    assert stored["intent"] == "translation"
    assert stored["verdict"] == VERDICT_UP
    assert stored["timestamp"]  # a non-empty ISO timestamp


def test_record_feedback_returns_the_record(tmp_path):
    record = record_feedback(
        "hello", "out_of_scope", VERDICT_DOWN, path=tmp_path / "f.jsonl"
    )

    assert isinstance(record, FeedbackRecord)
    assert record.verdict == VERDICT_DOWN
