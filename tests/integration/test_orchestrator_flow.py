from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
    build_orchestrator,
    process_message,
)


@patch("src.agents.orchestrator._build_classifier_chain")
async def test_knowledge_question_routes_to_rag_agent(mock_build_classifier):
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=INTENT_KNOWLEDGE_QUESTION)
    mock_build_classifier.return_value = mock_classifier

    mock_rag_chain = MagicMock()
    mock_rag_chain.ainvoke = AsyncMock(return_value="The White Card is required.")

    mock_translation_chain = MagicMock()
    mock_translation_chain.ainvoke = AsyncMock(return_value="should not be called")

    orchestrator = build_orchestrator(mock_rag_chain, mock_translation_chain)
    result = await process_message(orchestrator, "What is a White Card?")

    assert result == "The White Card is required."
    mock_rag_chain.ainvoke.assert_called_once()
    mock_translation_chain.ainvoke.assert_not_called()


@patch("src.agents.orchestrator._build_classifier_chain")
async def test_translation_request_routes_to_translation_agent(mock_build_classifier):
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=INTENT_TRANSLATION)
    mock_build_classifier.return_value = mock_classifier

    mock_rag_chain = MagicMock()
    mock_rag_chain.ainvoke = AsyncMock(return_value="should not be called")

    mock_translation_chain = MagicMock()
    mock_translation_chain.ainvoke = AsyncMock(return_value="Dobro jutro")

    orchestrator = build_orchestrator(mock_rag_chain, mock_translation_chain)
    result = await process_message(orchestrator, "Translate: Good morning")

    assert result == "Dobro jutro"
    mock_translation_chain.ainvoke.assert_called_once()
    mock_rag_chain.ainvoke.assert_not_called()


@patch("src.agents.orchestrator._build_classifier_chain")
async def test_unknown_intent_defaults_to_rag_agent(mock_build_classifier):
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value="something unexpected")
    mock_build_classifier.return_value = mock_classifier

    mock_rag_chain = MagicMock()
    mock_rag_chain.ainvoke = AsyncMock(return_value="RAG fallback response.")

    mock_translation_chain = MagicMock()

    orchestrator = build_orchestrator(mock_rag_chain, mock_translation_chain)
    result = await process_message(orchestrator, "Hello there")

    assert result == "RAG fallback response."
    mock_rag_chain.ainvoke.assert_called_once()
