from src.knowledge.loader import CHUNK_SIZE, chunk_documents, load_documents


def test_load_documents_reads_txt_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("First document content.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Second document content.", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert len(documents) == 2
    contents = {doc.page_content for doc in documents}
    assert "First document content." in contents
    assert "Second document content." in contents


def test_load_documents_ignores_non_txt_files(tmp_path):
    (tmp_path / "notes.txt").write_text("Valid file.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0].page_content == "Valid file."


def test_load_documents_includes_source_metadata(tmp_path):
    (tmp_path / "info.txt").write_text("Some info.", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert "source" in documents[0].metadata


def test_chunk_documents_splits_long_text(tmp_path):
    long_text = "This is a sentence about Serbian immigration. " * 100
    (tmp_path / "long.txt").write_text(long_text, encoding="utf-8")

    documents = load_documents(str(tmp_path))
    chunks = chunk_documents(documents)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + 50


def test_chunk_documents_preserves_metadata(tmp_path):
    long_text = "Immigration procedures in Serbia. " * 100
    (tmp_path / "procedures.txt").write_text(long_text, encoding="utf-8")

    documents = load_documents(str(tmp_path))
    chunks = chunk_documents(documents)

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "start_index" in chunk.metadata


def test_chunk_documents_keeps_short_text_intact(tmp_path):
    short_text = "Short document."
    (tmp_path / "short.txt").write_text(short_text, encoding="utf-8")

    documents = load_documents(str(tmp_path))
    chunks = chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
