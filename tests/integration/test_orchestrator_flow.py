from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
    OUT_OF_SCOPE_MESSAGE,
    build_orchestrator,
    process_message,
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
