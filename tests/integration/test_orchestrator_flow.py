from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agents.multimodal_agent import DEFAULT_QUESTION
from src.agents.orchestrator import (
    INTENT_DOCUMENT,
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
    MAX_HISTORY_MESSAGES,
    OUT_OF_SCOPE_MESSAGE,
    DocumentTurn,
    add_preference,
    build_orchestrator,
    clear_history,
    clear_preferences,
    get_history,
    get_preferences,
    process_document,
    process_message,
    remove_preference,
    tidy_preferences,
)

# Stands in for a photograph. Short enough to search a SQLite file for, so a
# test can prove the image never reached the checkpointer.
IMAGE_URL = "data:image/jpeg;base64,QUJDREVGR0hJSktMTU5PUFFSU1Q="


def _chains():
    """A mock RAG chain and translation chain with distinct responses."""
    rag = MagicMock()
    rag.ainvoke = AsyncMock(return_value="rag answer")
    translation = MagicMock()
    translation.ainvoke = AsyncMock(return_value="translation answer")
    return rag, translation


@contextmanager
def _document_agent(transcript="a bill", answer="It is an electricity bill."):
    """Stand in for the two chains build_orchestrator makes for itself.

    They are patched at the module rather than passed in, because the
    orchestrator builds them the same way it builds its classifier chain -
    neither needs anything but settings. The patch has to be active only
    while the graph is built, since the chains are captured there.

    Yields:
        The transcription chain and the summary chain, to assert against.
    """
    transcription = MagicMock()
    transcription.ainvoke = AsyncMock(return_value=transcript)
    summary = MagicMock()
    summary.ainvoke = AsyncMock(return_value=answer)
    with (
        patch(
            "src.agents.orchestrator.build_transcription_chain",
            return_value=transcription,
        ),
        patch("src.agents.orchestrator.build_summary_chain", return_value=summary),
    ):
        yield transcription, summary


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_knowledge_routes_to_rag(mock_classify):
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "What is a White Card?")
    assert result.response == "rag answer"
    translation.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_translation_routes_to_translation(mock_classify):
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "Как будет спасибо по-сербски")
    assert result.response == "translation answer"
    rag.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_confident_out_of_scope_is_rejected(mock_classify):
    mock_classify.return_value = (INTENT_OUT_OF_SCOPE, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    result = await process_message(orchestrator, "tell me a joke")
    assert result.response == OUT_OF_SCOPE_MESSAGE
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
    assert result.response == "translation answer"
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

    assert result.response == "translation answer"
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
async def test_add_preference_persists_and_returns_all_rules():
    # /pref add writes to the checkpointer; get_preferences reads it back. Two
    # different rules both survive, oldest first.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    await add_preference(orchestrator, "chat-add", "Write Serbian in Cyrillic.")
    rules = await add_preference(orchestrator, "chat-add", "Keep answers short.")
    assert rules == ["Write Serbian in Cyrillic.", "Keep answers short."]
    assert await get_preferences(orchestrator, "chat-add") == rules


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_add_preference_is_idempotent():
    # Adding the same rule twice does not duplicate it (the reducer dedupes).
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    await add_preference(orchestrator, "chat-dup", "Write Serbian in Cyrillic.")
    rules = await add_preference(orchestrator, "chat-dup", "Write Serbian in Cyrillic.")
    assert rules == ["Write Serbian in Cyrillic."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_get_preferences_is_empty_for_an_untouched_chat():
    # A chat that never set a preference (no checkpoint yet) yields an empty
    # list rather than raising.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    assert await get_preferences(orchestrator, "chat-none") == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_clear_preferences_empties_the_list():
    # /pref clear removes every stored rule for that chat.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    await add_preference(orchestrator, "chat-clear", "Write Serbian in Cyrillic.")
    await clear_preferences(orchestrator, "chat-clear")
    assert await get_preferences(orchestrator, "chat-clear") == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_command_set_preference_reaches_translation_agent(mock_classify):
    # End-to-end: a rule saved with /pref add is applied to a later translation
    # on the same chat, even though the message itself says nothing about it.
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-cmd-pref"
    await add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic script.")
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
    await add_preference(orchestrator, thread_id, "Keep answers short.")
    for turn in range(3):
        await process_message(orchestrator, f"question {turn}", thread_id=thread_id)
    assert await get_preferences(orchestrator, thread_id) == ["Keep answers short."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_remove_preference_drops_the_rule_at_that_index():
    # /pref remove deletes one rule by position and keeps the rest in order.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-remove"
    await add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    await add_preference(orchestrator, thread_id, "Keep answers short.")
    remaining = await remove_preference(orchestrator, thread_id, 0)
    assert remaining == ["Keep answers short."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_remove_preference_out_of_range_is_a_noop():
    # An index past the end leaves the stored list untouched.
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-remove-oob"
    await add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    remaining = await remove_preference(orchestrator, thread_id, 5)
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
    await add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")
    await add_preference(orchestrator, thread_id, "Add 5 examples per translation.")
    await add_preference(orchestrator, thread_id, "Give five examples per word.")

    result = await tidy_preferences(orchestrator, tidier, thread_id)

    assert result == ["Write Serbian in Cyrillic.", "Give 5 example sentences."]
    assert await get_preferences(orchestrator, thread_id) == result


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_a_document_turn_is_remembered_as_history(mock_classify):
    # The node writes the turn as ordinary messages, so the next text message
    # sees the photograph as context without anything being recorded by hand.
    mock_classify.return_value = (INTENT_TRANSLATION, 0.99)
    rag, translation = _chains()
    with _document_agent(transcript="Racun za grejanje"):
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-doc"
    await process_document(orchestrator, DocumentTurn(image_url=IMAGE_URL), thread_id)

    await process_message(orchestrator, "и когда платить?", thread_id=thread_id)

    history = translation.ainvoke.call_args.args[0]["history"]
    assert "Racun za grejanje" in history[0].content
    assert history[1].content == "It is an electricity bill."


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_text_after_a_photo_is_still_classified(mock_classify):
    # Modality is stored state, so a turn that did not overwrite it would be
    # routed by the previous one: the question after a photo would re-enter
    # the document node, with no photo in the run's context to read.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    with _document_agent():
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-doc-classify"
    await process_document(orchestrator, DocumentTurn(image_url=IMAGE_URL), thread_id)

    await process_message(orchestrator, "how do I pay it?", thread_id=thread_id)

    mock_classify.assert_called_once()
    rag.ainvoke.assert_called_once()


def _sqlite_orchestrator(path):
    """Build an orchestrator whose memory is a SQLite file at `path`."""
    rag, translation = _chains()
    saver = AsyncSqliteSaver(aiosqlite.connect(str(path)))
    return build_orchestrator(rag, translation, saver)


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_preferences_survive_a_restart_with_a_sqlite_checkpointer(tmp_path):
    # The point of the disk-backed checkpointer. A second orchestrator reading
    # the same file stands in for the process being redeployed: with the
    # in-memory saver the rule would be gone.
    database = tmp_path / "checkpoints.sqlite"
    before = _sqlite_orchestrator(database)
    await add_preference(before, "chat-restart", "Write Serbian in Cyrillic.")

    after = _sqlite_orchestrator(database)

    assert await get_preferences(after, "chat-restart") == [
        "Write Serbian in Cyrillic."
    ]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_memory_checkpointer_starts_empty_each_time(tmp_path):
    # The control for the test above: without a file behind it, a fresh
    # orchestrator remembers nothing, which is the behaviour being fixed.
    rag, translation = _chains()
    before = build_orchestrator(rag, translation)
    await add_preference(before, "chat-ram", "Write Serbian in Cyrillic.")

    after = build_orchestrator(rag, translation)

    assert await get_preferences(after, "chat-ram") == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_tidy_preferences_skips_the_llm_for_short_lists():
    # With fewer than two rules there is nothing to merge, so the tidier chain
    # is never called (no needless LLM cost).
    tidier = MagicMock()
    tidier.ainvoke = AsyncMock()
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-tidy-short"
    await add_preference(orchestrator, thread_id, "Write Serbian in Cyrillic.")

    result = await tidy_preferences(orchestrator, tidier, thread_id)

    assert result == ["Write Serbian in Cyrillic."]
    tidier.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_knowledge_agent_receives_the_conversation(mock_classify):
    # Without this the RAG agent answered every question as if it were the
    # first: a follow-up about a document the bot had just described came
    # back as "I don't have enough information to answer that question."
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    with _document_agent(transcript="nov-25: 18.179 kWh", answer="A power bill."):
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-rag-history"
    await process_document(orchestrator, DocumentTurn(image_url=IMAGE_URL), thread_id)

    await process_message(orchestrator, "what was november?", thread_id=thread_id)

    history = rag.ainvoke.call_args.args[0]["history"]
    assert "nov-25: 18.179 kWh" in history[0].content
    assert history[1].content == "A power bill."


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_knowledge_agent_searches_on_the_question_alone(mock_classify):
    # Retrieval must key off the latest message, not the whole conversation:
    # searching on the transcript of a bill would bury a short follow-up.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    with _document_agent(transcript="a long bill transcript"):
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-rag-input"
    await process_document(orchestrator, DocumentTurn(image_url=IMAGE_URL), thread_id)

    await process_message(orchestrator, "what was november?", thread_id=thread_id)

    assert rag.ainvoke.call_args.args[0]["input"] == "what was november?"


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_clear_history_forgets_the_conversation(mock_classify):
    # Deletion goes through RemoveMessage: the messages channel appends by
    # design, so writing an empty list would add nothing and change nothing.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-reset"
    await process_message(orchestrator, "first", thread_id=thread_id)

    assert await clear_history(orchestrator, thread_id) == 2

    state = orchestrator.get_state({"configurable": {"thread_id": thread_id}})
    assert state.values.get("messages", []) == []


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_clear_history_keeps_the_preferences(mock_classify):
    # History and preferences are separate channels; someone starting a new
    # topic should not silently lose the rules they set with /pref.
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-reset-pref"
    await add_preference(orchestrator, thread_id, "Keep answers short.")
    await process_message(orchestrator, "first", thread_id=thread_id)

    await clear_history(orchestrator, thread_id)

    assert await get_preferences(orchestrator, thread_id) == ["Keep answers short."]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_the_chat_still_works_after_being_cleared(mock_classify):
    mock_classify.return_value = (INTENT_KNOWLEDGE_QUESTION, 0.99)
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-reset-reuse"
    await process_message(orchestrator, "first", thread_id=thread_id)
    await clear_history(orchestrator, thread_id)

    result = await process_message(orchestrator, "second", thread_id=thread_id)

    assert result.response == "rag answer"


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_clear_history_on_an_untouched_chat_reports_nothing():
    rag, translation = _chains()
    orchestrator = build_orchestrator(rag, translation)

    assert await clear_history(orchestrator, "chat-never-spoke") == 0


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_a_photo_reaches_the_document_agent_without_being_classified(
    mock_classify,
):
    # The point of the entry edge. Modality comes free with the update, so a
    # photograph never pays for an inference to find out where it belongs -
    # and the classifier could not read it anyway.
    rag, translation = _chains()
    with _document_agent():
        orchestrator = build_orchestrator(rag, translation)

    result = await process_document(
        orchestrator, DocumentTurn(image_url=IMAGE_URL), "chat-photo"
    )

    assert result.intent == INTENT_DOCUMENT
    mock_classify.assert_not_called()
    rag.ainvoke.assert_not_called()
    translation.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
@patch("src.agents.orchestrator.classify")
async def test_writing_a_preference_runs_no_agent(mock_classify):
    # Making the entry edge conditional means update_state evaluates it, and
    # the /pref helpers write that way. The branch is only asked which task
    # would come next, so saving a rule must still cost nothing.
    rag, translation = _chains()
    with _document_agent() as (transcription, summary):
        orchestrator = build_orchestrator(rag, translation)

    await add_preference(orchestrator, "chat-pref-cost", "Reply in Cyrillic.")

    mock_classify.assert_not_called()
    rag.ainvoke.assert_not_called()
    transcription.ainvoke.assert_not_called()
    summary.ainvoke.assert_not_called()


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_the_document_node_answers_from_its_own_transcript():
    # The single source of truth. Reading the image a second time to write
    # the answer would give two accounts that disagree, and since only the
    # transcript is kept, the bot would contradict itself one turn later.
    rag, translation = _chains()
    with _document_agent(transcript="Adresa DRASKA 1") as (transcription, summary):
        orchestrator = build_orchestrator(rag, translation)

    await process_document(
        orchestrator, DocumentTurn(image_url=IMAGE_URL), "chat-transcript"
    )

    assert transcription.ainvoke.await_count == 1
    assert summary.ainvoke.call_args.args[0]["transcript"] == "Adresa DRASKA 1"
    assert "image_url" not in summary.ainvoke.call_args.args[0]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_a_caption_becomes_the_question_and_its_absence_a_default():
    rag, translation = _chains()
    with _document_agent() as (_, summary):
        orchestrator = build_orchestrator(rag, translation)

    await process_document(
        orchestrator, DocumentTurn(image_url=IMAGE_URL, caption="Kada?"), "chat-cap"
    )
    assert summary.ainvoke.call_args.args[0]["question"] == "Kada?"

    await process_document(
        orchestrator, DocumentTurn(image_url=IMAGE_URL), "chat-no-cap"
    )
    assert summary.ainvoke.call_args.args[0]["question"] == DEFAULT_QUESTION


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_the_document_turn_is_remembered_with_its_whole_transcript():
    # The user is shown a short answer; the page itself goes into memory, so
    # a later question can reach a figure the answer never mentioned.
    rag, translation = _chains()
    with _document_agent(transcript="Instalisana snaga 3,96", answer="A bill."):
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-doc-memory"

    photo = DocumentTurn(image_url=IMAGE_URL, caption="Sta je ovo?")
    result = await process_document(orchestrator, photo, thread_id)

    asked, answered = await get_history(orchestrator, thread_id)
    assert "Sta je ovo?" in asked.content
    assert "Instalisana snaga 3,96" in asked.content
    assert "Instalisana snaga 3,96" not in result.response
    assert answered.content == "A bill."


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_stored_preferences_reach_the_document_agent():
    # The node reads them straight out of the state it was handed, which is
    # why answering a photo no longer needs its own trip to the checkpointer.
    rag, translation = _chains()
    with _document_agent() as (_, summary):
        orchestrator = build_orchestrator(rag, translation)
    thread_id = "chat-doc-pref"
    await add_preference(orchestrator, thread_id, "Answer in Russian.")

    await process_document(orchestrator, DocumentTurn(image_url=IMAGE_URL), thread_id)

    assert "Answer in Russian." in summary.ainvoke.call_args.args[0]["preferences"]


@patch("src.agents.orchestrator.load_classifier", MagicMock())
async def test_the_photograph_never_reaches_the_checkpoint_file(tmp_path):
    # Why the image travels as runtime context instead of as state: every
    # state field is written to the checkpointer, and at MAX_IMAGE_BYTES a
    # base64 photo would add megabytes to this file on each upload. The
    # transcript is text and is meant to be there.
    path = tmp_path / "checkpoints.sqlite"
    rag, translation = _chains()
    saver = AsyncSqliteSaver(aiosqlite.connect(str(path)))
    with _document_agent(transcript="Racun 2.066,98 RSD"):
        orchestrator = build_orchestrator(rag, translation, saver)

    await process_document(
        orchestrator, DocumentTurn(image_url=IMAGE_URL), "chat-on-disk"
    )
    await saver.conn.close()

    written = path.read_bytes()
    assert IMAGE_URL.split(",")[1].encode() not in written
    assert b"2.066,98 RSD" in written
