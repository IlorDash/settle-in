from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(directory: str) -> list[Document]:
    """Load all .txt files from a directory into LangChain Document objects.

    Args:
        directory: Path to folder containing .txt files.

    Returns:
        List of Document objects, one per file. Each Document has
        `page_content` (the text) and `metadata` (source file path).
    """
    loader = DirectoryLoader(
        path=directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for vector search.

    Args:
        documents: List of full-size Document objects from load_documents().

    Returns:
        List of smaller Document objects. Each chunk preserves the original
        metadata (source file) plus its start index in the original text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
