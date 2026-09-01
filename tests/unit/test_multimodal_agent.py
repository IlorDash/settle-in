import base64
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.multimodal_agent import (
    IMAGE_DETAIL,
    SUMMARY_MODEL,
    TranscriptRequest,
    build_summary_chain,
    build_transcription_chain,
    encode_image_as_data_url,
    summarise,
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


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_transcription_prompt_fills_the_image_into_its_content_block():
    # The image variable sits inside a nested dict rather than in a plain
    # string. If LangChain ever stopped substituting there, the literal
    # "{image_url}" would be sent to the model and every read would fail.
    prompt = build_transcription_chain().steps[0]

    messages = prompt.invoke({"image_url": "data:image/jpeg;base64,AAAA"}).to_messages()

    image_block = messages[-1].content[0]
    assert image_block["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_transcription_prompt_carries_the_configured_image_detail():
    # Left to the API's "auto" default the photo is downscaled before the
    # model looks at it, and digits in a dense table come back misread.
    prompt = build_transcription_chain().steps[0]

    messages = prompt.invoke({"image_url": "data:image/jpeg;base64,AAAA"}).to_messages()

    assert messages[-1].content[0]["image_url"]["detail"] == IMAGE_DETAIL


@patch("src.agents.multimodal_agent.ChatOpenAI")
def test_the_summary_chain_uses_the_cheap_text_model(mock_llm):
    # It answers from text, so it needs no vision. Output is billed at
    # $0.60/1M here against $15.00/1M on the vision model, and the reply is
    # what output means.
    build_summary_chain()

    assert mock_llm.call_args.kwargs["model"] == SUMMARY_MODEL


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_summary_prompt_carries_the_transcript():
    prompt = build_summary_chain().steps[0]

    messages = prompt.invoke(
        {
            "transcript": "Instalisana snaga 3,96 kW",
            "question": "What is this?",
            "preferences": "(no special preferences)",
        }
    ).to_messages()

    assert "Instalisana snaga 3,96 kW" in messages[0].content


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_summary_prompt_puts_preferences_next_to_the_users_turn():
    # Same reasoning as the translation agent: a standing rule placed at the
    # top of the prompt is followed far less often than one placed here.
    prompt = build_summary_chain().steps[0]

    messages = prompt.invoke(
        {
            "transcript": "a bill",
            "question": "What is this?",
            "preferences": "- Reply in Russian.",
        }
    ).to_messages()

    assert messages[-2].content == "- Reply in Russian."


@patch("src.agents.multimodal_agent.ChatOpenAI", MagicMock())
def test_the_summary_prompt_works_without_preferences():
    # /pref is optional, so the partial default has to cover a chat that
    # never set a rule.
    prompt = build_summary_chain().steps[0]

    messages = prompt.invoke(
        {"transcript": "a bill", "question": "What is this?"}
    ).to_messages()

    assert messages[-2].content == "(no special preferences)"


async def test_summarise_returns_the_models_answer():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value="An electricity bill from EPS.")

    result = await summarise(
        chain, TranscriptRequest("Racun za struju", "What is this?", "(none)")
    )

    assert result == "An electricity bill from EPS."


async def test_summarise_passes_every_field_to_the_chain():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value="An electricity bill.")

    await summarise(
        chain, TranscriptRequest("Racun 2.066,98", "Kada plaćam?", "(none)")
    )

    assert chain.ainvoke.call_args.args[0] == {
        "transcript": "Racun 2.066,98",
        "question": "Kada plaćam?",
        "preferences": "(none)",
    }
