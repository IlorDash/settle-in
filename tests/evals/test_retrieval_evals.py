"""Check that a follow-up question still searches the right document.

These send REAL requests to OpenAI and cost money, so the `eval` marker keeps
them out of every ordinary run (see pyproject.toml). Run them with:

    pytest -m eval tests/evals/test_retrieval_evals.py -v

Each case is a real exchange that failed before `_build_search_query` was
added to the chain, and what it retrieved then is recorded next to it, since
these tests can no longer produce that state.

Three levels, deliberately graded. The first tests fire the follow-up through
the real chain with a spying retriever and check which knowledge-base files
the search reached. The second checks that the rewritten query is still a
question — a model that answers the follow-up instead of rewriting it puts
enough topic words in the query to retrieve correctly, so the first tests
alone cannot tell the two apart. The last checks the reply itself, against a
figure printed in the knowledge base that has to survive all the way to the
user. If the retrieval tests pass and the answer test fails, the search found
the page and the model failed to read it, which is a prompt problem rather
than a retrieval one.

One API call per distinct opening question: those answers are cached, so the
cases that share an opening pay for it once.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agents.rag_agent import build_rag_chain
from src.knowledge.vectorstore import get_retriever, load_vectorstore

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="needs a real OpenAI API key"
    ),
]


@dataclass(frozen=True)
class FollowUpCase:
    """One exchange whose second question cannot be understood on its own.

    Attributes:
        opening: The question that establishes the topic.
        follow_up: What the user actually types next.
        expected_source: The knowledge-base file that holds the answer.
    """

    opening: str
    follow_up: str
    expected_source: str


# Measured on 2026-08-25 with the retriever searching on the follow-up as
# typed, which is what the chain did before the rewrite step existed. Each
# comment is the four chunks that came back, in rank order.
CASES = [
    FollowUpCase(
        # was: 02 x3, 07 - the family document list of the wrong procedure
        opening="Какие документы нужны для белого картона?",
        follow_up="а для семьи?",
        expected_source="01_white_card_registration.txt",
    ),
    FollowUpCase(
        # was: 02 x3, 07 - deadlines, but the residence permit's, not the
        # 24 hours the white card is actually subject to
        opening="Какие документы нужны для белого картона?",
        follow_up="А сколько времени на это даётся?",
        expected_source="01_white_card_registration.txt",
    ),
    FollowUpCase(
        # was: 06 x3, 03 - electricity bills and bank charges, after which
        # the bot refused a fee that is printed in 02
        opening="Как получить временный вид на жительство в Сербии?",
        follow_up="А сколько это стоит?",
        expected_source="02_temporary_residence_permit.txt",
    ),
]


# 02_temporary_residence_permit.txt prints this as "approximately 20,660 RSD".
# The reply may group the digits with a dot, a comma or a space, and Russian
# usually uses the space, so the separator is matched rather than assumed.
FEE_PATTERN = re.compile(r"20[.,\s]?660")

# A rewritten question runs 60-80 characters here. Measured with the rewrite
# instruction placed first instead of last, the model answered the follow-up
# rather than rewriting it and returned 176-271 characters, so the ceiling
# separates the two without pinning any particular phrasing.
MAX_QUERY_CHARS = 120

# The one case whose answer is a single figure, which makes it the only one
# that can be graded on the reply instead of on the search.
FEE_CASE = next(
    case
    for case in CASES
    if case.expected_source == "02_temporary_residence_permit.txt"
)

_answers: dict[str, str] = {}


class Search:
    """What the chain sent to the retriever, and what came back.

    Attributes:
        queries: The text of each search, in the order they were run.
        sources: The knowledge-base file of every retrieved chunk.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.sources: list[str] = []


def spying_retriever(search: Search) -> RunnableLambda:
    """Wrap the real retriever so a test can see what it was asked for.

    Args:
        search: Recorder the query and the returned files are written to.

    Returns:
        A runnable the chain can search with, in place of the retriever.
    """
    retriever = get_retriever(load_vectorstore())

    def retrieve(query: str) -> list:
        documents = retriever.invoke(query)
        search.queries.append(query)
        print(f"\n  searched on: {query}")
        for document in documents:
            source = Path(document.metadata["source"]).name
            print(f"    -> {source}")
            search.sources.append(source)
        return documents

    return RunnableLambda(retrieve)


async def opening_exchange(case: FollowUpCase) -> list:
    """Ask a case's opening question for real and return the turn it produced.

    The follow-up is only ambiguous because of what came before it, so the
    history it is resolved against has to be the answer the bot actually
    gave, not a stand-in written here.

    Args:
        case: The exchange being probed.

    Returns:
        The opening question and its answer, as the orchestrator would pass
        them to the chain.
    """
    if case.opening not in _answers:
        chain = build_rag_chain(get_retriever(load_vectorstore()))
        _answers[case.opening] = await chain.ainvoke(
            {"input": case.opening, "history": []}
        )
    return [
        HumanMessage(content=case.opening),
        AIMessage(content=_answers[case.opening]),
    ]


async def follow_up_search(case: FollowUpCase) -> Search:
    """Run one case's follow-up through the chain and report what it searched.

    Args:
        case: The exchange being probed.

    Returns:
        The queries the chain searched on and the files they returned.
    """
    history = await opening_exchange(case)
    search = Search()
    chain = build_rag_chain(spying_retriever(search))

    await chain.ainvoke({"input": case.follow_up, "history": history})
    return search


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.follow_up)
async def test_a_follow_up_searches_the_document_holding_the_answer(
    case: FollowUpCase,
):
    search = await follow_up_search(case)

    assert case.expected_source in search.sources


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.follow_up)
async def test_a_follow_up_is_rewritten_into_a_question_not_an_answer(
    case: FollowUpCase,
):
    search = await follow_up_search(case)

    assert len(search.queries[0]) <= MAX_QUERY_CHARS


async def test_the_fee_follow_up_is_answered_with_the_fee():
    # Measured before history-aware retrieval existed: this follow-up sent
    # "А сколько это стоит?" to the search on its own, came back with
    # electricity bills, and the bot said it had no information about a fee
    # that is printed in the knowledge base.
    history = await opening_exchange(FEE_CASE)
    chain = build_rag_chain(get_retriever(load_vectorstore()))

    answer = await chain.ainvoke({"input": FEE_CASE.follow_up, "history": history})

    print(f"\n----- {FEE_CASE.follow_up} -----\n{answer}")
    assert FEE_PATTERN.search(answer)
