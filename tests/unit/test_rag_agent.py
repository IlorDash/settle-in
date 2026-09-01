from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agents.rag_agent import ask, build_rag_chain

# What the fake LLM below returns for any rewrite call, so a test can tell
# a rewritten query apart from the raw question it was given.
REWRITTEN_QUERY = "Rewritten standalone question?"


def _spying_retriever(queries: list) -> RunnableLambda:
    """A fake retriever that records the query text it is searched on.

    A bare MagicMock cannot sit in an LCEL pipe, so this is a real
    RunnableLambda standing in for the VectorStoreRetriever.
    """

    def retrieve(query: str) -> list[Document]:
        queries.append(query)
        return [Document(page_content="context chunk")]

    return RunnableLambda(retrieve)


def _counting_llm(calls: list) -> RunnableLambda:
    """A fake ChatOpenAI that records every call and always rewrites the
    same way, so a test can both count LLM invocations and recognise a
    rewritten query by its fixed content.
    """

    def respond(prompt_value) -> AIMessage:
        calls.append(prompt_value)
        return AIMessage(content=REWRITTEN_QUERY)

    return RunnableLambda(respond)


@patch("src.agents.rag_agent.ChatOpenAI")
def test_build_rag_chain_returns_runnable(mock_llm):
    mock_retriever = MagicMock()

    chain = build_rag_chain(mock_retriever)

    assert chain is not None
    mock_llm.assert_called_once()


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_missing_history_key_searches_on_the_raw_question(mock_llm_class):
    queries: list = []
    mock_llm_class.return_value = _counting_llm([])
    chain = build_rag_chain(_spying_retriever(queries))

    await chain.ainvoke({"input": "What is a White Card?"})

    assert queries == ["What is a White Card?"]


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_empty_history_searches_on_the_raw_question(mock_llm_class):
    queries: list = []
    mock_llm_class.return_value = _counting_llm([])
    chain = build_rag_chain(_spying_retriever(queries))

    await chain.ainvoke({"input": "What is a White Card?", "history": []})

    assert queries == ["What is a White Card?"]


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_no_history_does_not_call_the_llm_to_rewrite(mock_llm_class):
    calls: list = []
    mock_llm_class.return_value = _counting_llm(calls)
    chain = build_rag_chain(_spying_retriever([]))

    await chain.ainvoke({"input": "What is a White Card?"})

    # The one call left is the final answer; a rewrite call would make it two.
    assert len(calls) == 1


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_history_present_searches_on_the_rewritten_query(mock_llm_class):
    queries: list = []
    mock_llm_class.return_value = _counting_llm([])
    chain = build_rag_chain(_spying_retriever(queries))
    history = [
        HumanMessage(content="What is a White Card?"),
        AIMessage(content="It is a proof of residence registration."),
    ]

    await chain.ainvoke({"input": "How much does it cost?", "history": history})

    assert queries == [REWRITTEN_QUERY]


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_history_present_calls_the_llm_twice(mock_llm_class):
    calls: list = []
    mock_llm_class.return_value = _counting_llm(calls)
    chain = build_rag_chain(_spying_retriever([]))
    history = [
        HumanMessage(content="What is a White Card?"),
        AIMessage(content="It is a proof of residence registration."),
    ]

    await chain.ainvoke({"input": "How much does it cost?", "history": history})

    # One call rewrites the query, the other produces the final answer.
    assert len(calls) == 2


@patch("src.agents.rag_agent.ChatOpenAI")
async def test_ask_completes_when_the_chain_receives_no_history_key(mock_llm_class):
    mock_llm_class.return_value = _counting_llm([])
    chain = build_rag_chain(_spying_retriever([]))

    answer = await ask(chain, "What is a White Card?")

    assert answer == REWRITTEN_QUERY


async def test_ask_returns_answer_from_chain():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="test answer")

    result = await ask(mock_chain, "test question")

    assert result == "test answer"
    mock_chain.ainvoke.assert_called_once_with({"input": "test question"})
