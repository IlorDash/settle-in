from langchain_core.messages import AIMessage, HumanMessage

from src.bot.transcript import format_transcript, speaker


def test_speaker_labels_the_bot_side():
    assert speaker(AIMessage(content="hello")).strip() == "bot"


def test_speaker_labels_everything_else_as_the_user():
    assert speaker(HumanMessage(content="hello")).strip() == "user"


def test_format_transcript_keeps_only_the_last_messages():
    messages = [HumanMessage(content=str(number)) for number in range(1, 6)]

    lines = format_transcript(messages, limit=2).splitlines()

    assert [line.split(": ")[-1] for line in lines] == ["4", "5"]


def test_format_transcript_numbers_from_the_start_of_the_history():
    # Numbering the kept slice from 1 would make two exports of the same chat
    # impossible to line up; these are positions in the whole history.
    messages = [HumanMessage(content=str(number)) for number in range(1, 6)]

    lines = format_transcript(messages, limit=2).splitlines()

    assert lines[0].startswith("[4]")


def test_format_transcript_keeps_everything_when_the_limit_is_zero():
    messages = [HumanMessage(content=str(number)) for number in range(1, 6)]

    assert len(format_transcript(messages, limit=0).splitlines()) == 5


def test_format_transcript_marks_who_said_what():
    messages = [HumanMessage(content="question"), AIMessage(content="answer")]

    lines = format_transcript(messages, limit=2).splitlines()

    assert lines[0].endswith("user: question")
    assert lines[1].endswith("bot : answer")
