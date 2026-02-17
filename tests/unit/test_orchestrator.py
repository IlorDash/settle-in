from unittest.mock import AsyncMock, MagicMock

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
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


async def test_process_message_passes_user_message():
    mock_orchestrator = MagicMock()
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={
            "user_message": "Translate: hello",
            "intent": INTENT_TRANSLATION,
            "agent_response": "Zdravo",
        }
    )

    await process_message(mock_orchestrator, "Translate: hello")

    mock_orchestrator.ainvoke.assert_called_once_with(
        {"user_message": "Translate: hello"}
    )


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
