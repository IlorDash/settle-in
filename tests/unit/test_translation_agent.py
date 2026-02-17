from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.translation_agent import build_translation_chain, translate


@patch("src.agents.translation_agent.ChatOpenAI")
def test_build_translation_chain_returns_runnable(mock_llm):
    chain = build_translation_chain()

    assert chain is not None
    mock_llm.assert_called_once()


async def test_translate_returns_translated_text():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="Dobro jutro")

    result = await translate(mock_chain, "Good morning")

    assert result == "Dobro jutro"


async def test_translate_passes_text_as_input():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="Hello")

    await translate(mock_chain, "Zdravo")

    mock_chain.ainvoke.assert_called_once_with({"input": "Zdravo"})
