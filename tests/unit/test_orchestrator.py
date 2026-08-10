from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
    MAX_STORED_MESSAGES,
    MAX_STORED_PREFERENCES,
    _parse_rule_lines,
    _preferences_directive,
    append_capped_messages,
    merge_preferences,
    process_message,
)


async def test_process_message_returns_agent_response():
    mock_orchestrator = MagicMock()
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={
            "user_message": "What is a White Card?",
            "intent": INTENT_KNOWLEDGE_QUESTION,
            "agent_response": "The White Card is a registration document.",
        }
    )

    result = await process_message(mock_orchestrator, "What is a White Card?")

    assert result == "The White Card is a registration document."


async def test_process_message_returns_translation_response():
    mock_orchestrator = MagicMock()
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={
            "user_message": "Dobro jutro",
            "intent": INTENT_TRANSLATION,
            "agent_response": "Good morning",
        }
    )

    result = await process_message(mock_orchestrator, "Dobro jutro")

    assert result == "Good morning"


def test_append_capped_messages_bounds_the_log():
    # Adding one message to a full log drops the oldest, keeping the newest.
    existing = [HumanMessage(content=str(i)) for i in range(MAX_STORED_MESSAGES)]

    result = append_capped_messages(existing, [HumanMessage(content="newest")])

    assert len(result) == MAX_STORED_MESSAGES
    assert result[-1].content == "newest"
    assert result[0].content == "1"


def test_merge_preferences_replaces_with_reconciled_list():
    # The reconciled list replaces the stored one (the node already merged old
    # and new), and duplicates within it are dropped.
    existing = {"instructions": ["Write in Cyrillic."]}
    merged = merge_preferences(existing, {"instructions": ["Write in Latin."]})
    assert merged["instructions"] == ["Write in Latin."]

    deduped = merge_preferences({}, {"instructions": ["A", "A", "B"]})
    assert deduped["instructions"] == ["A", "B"]


def test_merge_preferences_caps_stored_instructions():
    # An over-long reconciled list is bounded, keeping the most recent rules.
    over = {"instructions": [f"rule {i}" for i in range(MAX_STORED_PREFERENCES + 5)]}

    merged = merge_preferences({}, over)

    assert len(merged["instructions"]) == MAX_STORED_PREFERENCES
    assert merged["instructions"][-1] == f"rule {MAX_STORED_PREFERENCES + 4}"
    assert "rule 0" not in merged["instructions"]


def test_merge_preferences_replaces_scalar_values():
    # Non-list values (e.g. a future scalar preference) are replaced outright,
    # not merged like the "instructions" list.
    merged = merge_preferences({"tone": "formal"}, {"tone": "casual"})

    assert merged["tone"] == "casual"


def test_preferences_directive_returns_placeholder_when_no_preferences():
    # With nothing stored yet, the agent gets an explicit "none" directive
    # rather than an empty or malformed instruction block.
    assert _preferences_directive({}) == "(no special preferences)"


def test_preferences_directive_returns_placeholder_when_preferences_none():
    # `preferences` itself can be None (e.g. a brand-new thread); this must
    # not raise and must fall back to the same placeholder.
    assert _preferences_directive(None) == "(no special preferences)"


def test_preferences_directive_lists_stored_instructions():
    # Stored instructions are rendered as a bulleted directive the
    # translation agent's prompt can follow.
    preferences = {"instructions": ["Reply in Cyrillic.", "Keep answers short."]}

    directive = _preferences_directive(preferences)

    assert directive == (
        "Apply these standing user preferences to every reply, whatever the "
        "target language:\n"
        "- Reply in Cyrillic.\n"
        "- Keep answers short."
    )


def test_parse_rule_lines_strips_markers_but_keeps_number_rules():
    # The tidier may return bullets, "1." numbering, or blank lines; those are
    # cleaned off. A rule that merely starts with a number ("5 examples...")
    # has no trailing "." or ")", so it must be left intact, not truncated.
    text = "- Reply in Cyrillic.\n2. Keep answers short.\n\n5 examples per word"

    result = _parse_rule_lines(text)

    assert result == [
        "Reply in Cyrillic.",
        "Keep answers short.",
        "5 examples per word",
    ]
