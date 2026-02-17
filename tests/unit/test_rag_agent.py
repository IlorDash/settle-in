from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.rag_agent import ask, build_rag_chain


@patch("src.agents.rag_agent.ChatOpenAI")
def test_build_rag_chain_returns_runnable(mock_llm):
    mock_retriever = MagicMock()

    chain = build_rag_chain(mock_retriever)

    assert chain is not None
    mock_llm.assert_called_once()


async def test_ask_returns_answer_from_chain():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="test answer")

    result = await ask(mock_chain, "test question")

    assert result == "test answer"
    mock_chain.ainvoke.assert_called_once_with("test question")


async def test_ask_passes_query_as_string():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="Apply at the embassy.")

    await ask(mock_chain, "How to get a visa?")

    mock_chain.ainvoke.assert_called_once_with("How to get a visa?")
