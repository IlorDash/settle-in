from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
    MAX_HISTORY_MESSAGES,
    OUT_OF_SCOPE_MESSAGE,
    add_preference,
    build_orchestrator,
    clear_preferences,
    get_preferences,
    process_message,
    remove_preference,
    tidy_preferences,
)


def _chains():
    """A mock RAG chain and translation chain with distinct responses."""
    rag = MagicMock()
    rag.ainvoke = AsyncMock(return_value="rag answer")
    translation = MagicMock()
    translation.ainvoke = AsyncMock(return_value="translation answer")
    return rag, translation


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_knowledge_routes_to_rag(mock_classify):
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "What is a White Card?")
    assert result == "rag answer"
    translation.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_translation_routes_to_translation(mock_classify):
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "Как будет спасибо по-сербски")
    assert result == "translation answer"
    rag.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_out_of_scope_is_rejected(mock_classify):
    mock_classify.return_value = (INTENT_OUT_OF_SCOPE, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "tell me a joke")
    assert result == OUT_OF_SCOPE_MESSAGE
    rag.ainvoke.assert_not_called()
    translation.ainvoke.assert_not_called()


@patch("src.agents.orchestrator._build_classifier_chain")
@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_low_confidence_falls_back_to_the_llm(mock_classify, mock_build_chain):
    # DNN is unsure (below threshold); the LLM fallback decides the intent.
    mock_classify.return_value = (INTENT_OUT_OF_SCOPE, 0.2)
    llm_chain = MagicMock()
    llm_chain.ainvoke = AsyncMock(return_value="translation")
    mock_build_chain.return_value = llm_chain
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "ambiguous message")
    assert result == "translation answer"
    llm_chain.ainvoke.assert_awaited_once()


@patch("src.agents.orchestrator._build_classifier_chain")
@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_followup_escalates_to_context_aware_llm(mock_classify, mock_build_chain):
    # The DNN reads a bare "А латиницей?" as out_of_scope with high confidence,
    # but prior history marks it as a likely follow-up, so we re-check with the
    # context-aware LLM classifier, which routes it to translation.
    mock_classify.return_value = (INTENT_OUT_OF_SCOPE, 0.99)
    llm_chain = MagicMock()
    llm_chain.ainvoke = AsyncMock(return_value="translation")
    mock_build_chain.return_value = llm_chain
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-follow"
    await process_message(orchestrator, "Переведи привет", thread_id=thread_id)
    result = await process_message(orchestrator, "А латиницей?", thread_id=thread_id)

    assert result == "translation answer"
    # The LLM classifier was consulted with the earlier turns as context.
    payload = llm_chain.ainvoke.call_args.args[0]
    assert len(payload["history"]) == 2


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_same_thread_accumulates_history(mock_classify):
    # Two turns on one thread_id should leave 4 messages remembered:
    # user, bot, user, bot.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    config = {"configurable": {"thread_id": "chat-1"}}
    await process_message(orchestrator, "first question", thread_id="chat-1")
    await process_message(orchestrator, "second question", thread_id="chat-1")
    state = orchestrator.get_state(config)
    assert len(state.values["messages"]) == 4


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_separate_threads_do_not_share_history(mock_classify):
    # A message on one thread must not appear in another chat's memory.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    await process_message(orchestrator, "hello from A", thread_id="chat-A")
    state_b = orchestrator.get_state({"configurable": {"thread_id": "chat-B"}})
    assert state_b.values.get("messages", []) == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_history_passed_to_agent_is_bounded(mock_classify):
    # A long conversation must not send an ever-growing history to the agents.
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-bound"
    for turn in range(MAX_HISTORY_MESSAGES + 2):
        await process_message(orchestrator, f"message {turn}", thread_id=thread_id)
    history = translation.ainvoke.call_args.args[0]["history"]
    assert len(history) <= MAX_HISTORY_MESSAGES


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_translation_receives_stored_preference(mock_classify):
    # A stored standing instruction must reach the translation agent, so it
    # survives beyond the bounded history window.
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    await orchestrator.ainvoke(
        {
            "messages": [HumanMessage(content="как будет вилка")],
            "preferences": {"instructions": ["Write Serbian in Cyrillic script."]},
        },
        config={"configurable": {"thread_id": "chat-pref"}},
    )
    directive = translation.ainvoke.call_args.args[0]["preferences"]
    assert "Cyrillic" in directive


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_add_preference_persists_and_returns_all_rules():
    # /pref add writes to the checkpointer; get_preferences reads it back. Two
    # different rules both survive, oldest first.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    add_preference(orchestrator, "chat-add", "Write Serbian in Cyrillic.")
    rules = add_preference(orchestrator, "chat-add", "Keep answers short.")
    assert rules == ["Write Serbian in Cyrillic.", "Keep answers short."]
    assert get_preferences(orchestrator, "chat-add") == rules


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_add_preference_is_idempotent():
    # Adding the same rule twice does not duplicate it (the reducer dedupes).
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    add_preference(orchestrator, "chat-dup", "Write Serbian in Cyrillic.")
    rules = add_preference(orchestrator, "chat-dup", "Write Serbian in Cyrillic.")
    assert rules == ["Write Serbian in Cyrillic."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_get_preferences_is_empty_for_an_untouched_chat():
    # A chat that never set a preference (no checkpoint yet) yields an empty
    # list rather than raising.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    assert get_preferences(orchestrator, "chat-none") == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_clear_preferences_empties_the_list():
    # /pref clear removes every stored rule for that chat.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    add_preference(orchestrator, "chat-clear", "Write Serbian in Cyrillic.")
    clear_preferences(orchestrator, "chat-clear")
    assert get_preferences(orchestrator, "chat-clear") == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_command_set_preference_reaches_translation_agent(mock_classify):
    # End-to-end: a rule saved with /pref add is applied to a later translation
    # on the same chat, even though the message itself says nothing about it.
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-cmd-pref"
    add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic script.")
    await process_message(orchestrator, "как будет вилка", thread_id=thread_id)
    directive = translation.ainvoke.call_args.args[0]["preferences"]
    assert "Cyrillic" in directive


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_preferences_survive_across_normal_turns(mock_classify):
    # A preference set before any conversation is still present after several
    # unrelated turns (it lives in the same per-chat checkpointer state).
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-persist"
    add_preference(orchestrator, thread_id, "Keep answers short.")
    for turn in range(3):
        await process_message(orchestrator, f"question {turn}", thread_id=thread_id)
    assert get_preferences(orchestrator, thread_id) == ["Keep answers short."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_remove_preference_drops_the_rule_at_that_index():
    # /pref remove deletes one rule by position and keeps the rest in order.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-remove"
    add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    add_preference(orchestrator, thread_id, "Keep answers short.")
    remaining = remove_preference(orchestrator, thread_id, 0)
    assert remaining == ["Keep answers short."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
def test_remove_preference_out_of_range_is_a_noop():
    # An index past the end leaves the stored list untouched.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-remove-oob"
    add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    remaining = remove_preference(orchestrator, thread_id, 5)
    assert remaining == ["Write Serbian in Cyrillic."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_tidy_preferences_replaces_list_with_merged_rules():
    # /pref tidy asks the tidier to merge duplicates and stores its cleaned list.
    tidier = MagicMock()
    tidier.ainvoke = AsyncMock(
        return_value="Write Serbian in Cyrillic.\nGive 5 example sentences."
    )
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-tidy"
    add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    add_preference(orchestrator, thread_id, "Add 5 examples for each translation.")
    add_preference(orchestrator, thread_id, "Give five example sentences per word.")

    result = await tidy_preferences(orchestrator, tidier, thread_id)

    assert result == ["Write Serbian in Cyrillic.", "Give 5 example sentences."]
    assert get_preferences(orchestrator, thread_id) == result


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_tidy_preferences_skips_the_llm_for_short_lists():
    # With fewer than two rules there is nothing to merge, so the tidier chain
    # is never called (no needless LLM cost).
    tidier = MagicMock()
    tidier.ainvoke = AsyncMock()
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-tidy-short"
    add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")

    result = await tidy_preferences(orchestrator, tidier, thread_id)

    assert result == ["Write Serbian in Cyrillic."]
    tidier.ainvoke.assert_not_called()
