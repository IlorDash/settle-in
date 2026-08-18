import base64
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.multimodal_agent import (
    IMAGE_DETAIL,
    DocumentRequest,
    analyze_document,
    build_multimodal_chain,
    encode_image_as_data_url,
)


def test_encode_image_as_data_url_wraps_the_bytes():
    url = encode_image_as_data_url(b"pretend jpeg", "image/jpeg")

    assert url == "data:image/jpeg;base64," + base64.b64encode(b"pretend jpeg").decode(
        "ascii"
    )


def test_encode_image_as_data_url_keeps_the_media_type():
    # A PNG announced as a JPEG is rejected by the API, so the type travels.
    url = encode_image_as_data_url(b"pretend png", "image/png")

    assert url.startswith("data:image/png;base64,")


@patch("src.agents.multimodal_agent.ChatOpenAI")
def test_build_multimodal_chain_returns_runnable(mock_llm):
    chain = build_multimodal_chain()

    assert chain is not None
    mock_llm.assert_called_once()


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_prompt_fills_the_image_into_its_content_block():
    # The image variable sits inside a nested dict rather than in a plain
    # string. If LangChain ever stopped substituting there, the literal
    # "{image_url}" would be sent to the model and every read would fail.
    prompt = build_multimodal_chain().steps[0]

    messages = prompt.invoke(
        {
            "image_url": "data:image/jpeg;base64,AAAA",
            "question": "What is this?",
            "preferences": "(no special preferences)",
        }
    ).to_messages()

    image_block = messages[-1].content[1]
    assert image_block["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_prompt_carries_the_configured_image_detail():
    # Left to the API's "auto" default the photo is downscaled before the
    # model looks at it, and digits in a dense table come back misread. The
    # value is compared against the constant rather than a literal, so tuning
    # it stays a one-line change; what is pinned is that the key survives
    # into the nested content block at all.
    prompt = build_multimodal_chain().steps[0]

    messages = prompt.invoke(
        {
            "image_url": "data:image/jpeg;base64,AAAA",
            "question": "What is this?",
            "preferences": "(no special preferences)",
        }
    ).to_messages()

    assert messages[-1].content[1]["image_url"]["detail"] == IMAGE_DETAIL


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_prompt_puts_preferences_next_to_the_users_turn():
    # Same reasoning as the translation agent: a standing rule placed at the
    # top of the prompt is followed far less often than one placed here.
    prompt = build_multimodal_chain().steps[0]

    messages = prompt.invoke(
        {
            "image_url": "data:image/jpeg;base64,AAAA",
            "question": "What is this?",
            "preferences": "- Reply in Russian.",
        }
    ).to_messages()

    assert messages[-2].content == "- Reply in Russian."


async def test_analyze_document_returns_the_models_answer():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value="An electricity bill from EPS.")

    result = await analyze_document(
        chain, DocumentRequest("data:image/jpeg;base64,AAAA", "What is this?", "(none)")
    )

    assert result == "An electricity bill from EPS."


async def test_analyze_document_passes_every_field_to_the_chain():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value="An electricity bill.")

    await analyze_document(
        chain, DocumentRequest("data:image/png;base64,BBBB", "Kada plaćam?", "(none)")
    )

    assert chain.ainvoke.call_args.args[0] == {
        "image_url": "data:image/png;base64,BBBB",
        "question": "Kada plaćam?",
        "preferences": "(none)",
    }
