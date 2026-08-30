from langchain_core.documents import Document

from scripts.evaluate_retrieval import Passage, Query, Result, basename, retrieve

# A source string as Chroma stored it when the index was built on Windows.
WINDOWS_SOURCE = r"data\knowledge_base\01_white_card_registration.txt"
POSIX_SOURCE = "data/knowledge_base/01_white_card_registration.txt"
EXPECTED = "01_white_card_registration.txt"


class StubRetriever:
    """Returns the documents it was given, the way a retriever would."""

    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return self.documents


def a_query(document=EXPECTED):
    return Query(
        id=1,
        language="en",
        kind="plain",
        document=document,
        text="do i have to register my address?",
        expected_fact="within 24 hours",
    )


def test_basename_reads_a_windows_source_on_any_platform():
    # The Raspberry Pi run scored C1 at 0.0000 because pathlib.Path on Linux
    # does not split a Windows path, so every source compared as a whole
    # string and no expected document ever matched.
    assert basename(WINDOWS_SOURCE) == EXPECTED


def test_basename_reads_a_posix_source():
    assert basename(POSIX_SOURCE) == EXPECTED


def test_basename_leaves_a_bare_file_name_alone():
    assert basename(EXPECTED) == EXPECTED


def test_basename_passes_the_placeholder_for_missing_metadata_through():
    # retrieve() falls back to "?" when a chunk carries no source at all.
    assert basename("?") == "?"


def test_retrieve_names_a_windows_built_index_by_file_name():
    documents = [
        Document(
            page_content="Register within 24 hours of arrival.",
            metadata={"source": WINDOWS_SOURCE, "start_index": 0},
        )
    ]

    passages, elapsed = retrieve(StubRetriever(documents), a_query())

    assert [p.source for p in passages] == [EXPECTED]
    assert elapsed >= 0


def test_a_windows_built_index_scores_as_a_hit():
    # The property the score depends on: the same run must count as a hit
    # whatever machine reads the index back.
    documents = [
        Document(
            page_content="Register within 24 hours of arrival.",
            metadata={"source": WINDOWS_SOURCE, "start_index": 0},
        )
    ]
    query = a_query()

    passages, _ = retrieve(StubRetriever(documents), query)
    result = Result(query, passages, answer=None, retrieval_ms=1.0)

    assert result.hit
    assert result.rank == 1


def test_a_passage_from_another_document_is_not_a_hit():
    passages = [Passage(source="07_visa_regime_entry.txt", start_index=0, text="x")]

    result = Result(a_query(), passages, answer=None, retrieval_ms=1.0)

    assert not result.hit
    assert result.rank is None
