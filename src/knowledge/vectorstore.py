from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from src.config import settings

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVER_TOP_K = 4


def _create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=settings.openai_api_key,
    )


def build_vectorstore(document_chunks: list[Document]) -> Chroma:
    """Create a new ChromaDB collection from document chunks.

    Embeds all chunks using OpenAI and persists the database to disk.
    This is a one-time operation — call it when documents change.

    Args:
        document_chunks: Pre-chunked Document objects from loader.chunk_documents().

    Returns:
        Chroma vector store instance with all chunks embedded and stored.
    """
    return Chroma.from_documents(
        documents=document_chunks,
        embedding=_create_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=settings.chroma_persist_dir,
    )


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Returns:
        Chroma vector store instance loaded from settings.chroma_persist_dir.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_create_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """Create a retriever that finds the most relevant document chunks.

    Args:
        vectorstore: Chroma instance to search in.

    Returns:
        Retriever configured to return the top RETRIEVER_TOP_K results.
    """
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
