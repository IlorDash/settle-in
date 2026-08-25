from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are an assistant helping immigrants in Serbia. "
    "Answer using ONLY two things: the context below, and what has already "
    "been said in this conversation. The earlier turns count as much as the "
    "context does, so a question about something you have already explained "
    "is answerable from them, including arithmetic over figures you quoted. "
    "The context is a fixed number of closest matches, so some or all of "
    "it can be unrelated; ignore what does not help. "
    "If neither holds enough information, "
    'say "I don\'t have enough information to answer that question." '
    "Be concise and helpful.\n\n"
    "Context:\n{context}"
)

REWRITE_PROMPT = (
    "Rewrite the user's last message so it can be understood on its own, "
    "using the conversation above to fill in what the message "
    "leaves out. Add only the topic the message is about; do not carry "
    "over figures, dates or reference numbers. Keep the language it is "
    "written in. Do not answer it and do not add anything the "
    "conversation does not say. If it already stands on its own, return "
    "it unchanged. Return only the rewritten message."
)


def _format_documents(documents: list[Document]) -> str:
    """Join retrieved document chunks into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def _build_search_query(llm: ChatOpenAI) -> Runnable:
    """Build the step that decides what text the retriever searches on.

    A follow-up like "а для семьи?" carries none of the words its answer
    is filed under, so searching on it lands in an unrelated document.
    Rewriting it against the conversation puts the topic back in.

    Args:
        llm: The model that does the rewriting.

    Returns:
        A Runnable taking the chain's input and returning the query text.
    """
    # The rewrite stays in the language the user wrote in, although the
    # knowledge base is English. The question that opens a topic is never
    # rewritten, so the search is cross-lingual on that turn either way,
    # and an English follow-up would match on a different footing from the
    # question it follows.
    #
    # The instruction goes last, after the message it applies to, the way
    # the translation agent places its preferences. Measured: led with it,
    # the model answered "А сколько это стоит?" in a 176-character
    # paragraph instead of rewriting it; trailing, the same prompt
    # returned "Сколько стоит получение временного вида на жительство в
    # Сербии?".
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("history"),
            ("human", "{input}"),
            ("system", REWRITE_PROMPT),
        ]
    )
    # The first question of a chat has no conversation to be resolved
    # against and already names its own topic, so it goes to the search
    # word for word. Every later one pays an extra gpt-4o-mini call
    # before the answer can start.
    return RunnableBranch(
        (lambda inputs: not inputs.get("history"), itemgetter("input")),
        prompt | llm | StrOutputParser(),
    )


def build_rag_chain(retriever: VectorStoreRetriever):
    """Build a retrieval-augmented generation chain using LCEL.

    The chain retrieves relevant document chunks from the vector store,
    formats them into the prompt as context, and asks the LLM to answer.

    Args:
        retriever: Vector store retriever that finds relevant document chunks.

    Returns:
        A Runnable accepting {"input": <question>, "history": [...]} that
        returns the answer as a string. The history is optional.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
        ]
    )

    # `assign` adds the retrieved context while passing every other key
    # through, so history reaches the prompt without being named here. The
    # search runs on one question rather than on the whole conversation,
    # which would drown a short follow-up in the words of earlier turns.
    return (
        RunnablePassthrough.assign(
            context=_build_search_query(llm) | retriever | _format_documents
        )
        | prompt
        | llm
        | StrOutputParser()
    )


async def ask(chain, query: str) -> str:
    """Send a question through the RAG chain and return the answer.

    Args:
        chain: RAG chain built by build_rag_chain().
        query: User's question as a string.

    Returns:
        The LLM's answer grounded in the retrieved documents.
    """
    return await chain.ainvoke({"input": query})
