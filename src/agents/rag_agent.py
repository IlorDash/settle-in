from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
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
    "The context is retrieved by matching the latest message alone, so on a "
    "follow-up it is often unrelated; ignore it when it is. "
    "If neither holds enough information, "
    'say "I don\'t have enough information to answer that question." '
    "Be concise and helpful.\n\n"
    "Context:\n{context}"
)


def _format_documents(documents: list[Document]) -> str:
    """Join retrieved document chunks into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


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
    # through, so history reaches the prompt without being named here. Only
    # the question is used to search: retrieving on the whole conversation
    # would drown a short follow-up in the words of earlier turns.
    return (
        RunnablePassthrough.assign(
            context=itemgetter("input") | retriever | _format_documents
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
