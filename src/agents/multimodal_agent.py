import base64
from typing import NamedTuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

# The oldest model accepting detail="original", at the same $2.50/1M input
# tokens as gpt-4o. Dense tables are unreadable without that setting.
LLM_MODEL = "gpt-5.4"
LLM_TEMPERATURE = 0

# The reply is written from the transcript, so it needs no vision and no
# strength beyond reading text: gpt-4o-mini charges $0.60/1M for output
# against $15.00/1M on the vision model, and output is what a reply costs.
SUMMARY_MODEL = "gpt-4o-mini"

# Sends the image at its own resolution, budgeted in 32x32 patches up to
# roughly 10 megapixels, instead of the 768px shortest side gpt-4o-class
# models impose. Only gpt-5.4 and newer accept it, so this and LLM_MODEL move
# together; tests/evals/test_vision_evals.py scores any pairing.
IMAGE_DETAIL = "original"

# A dense bill transcribes to roughly 2000 tokens. The cap is headroom: a
# model that cannot read a table repeats one line until something stops it.
TRANSCRIPTION_MAX_TOKENS = 4000

# What to ask the model when the user sent a photo with no caption. The chat
# remembers the transcript rather than this answer, so leaving something out
# of it loses nothing: detail is one follow-up away. Asking instead for every
# figure on the page produced replies past Telegram's own length limit.
DEFAULT_QUESTION = (
    "What is this, and what does it say? Keep it to a short summary - I will "
    "ask if I want detail."
)

ANSWER_PROMPT = (
    "You are an assistant for immigrants in Serbia. Below is a verbatim "
    "transcript of a photo the user sent, usually an official document, "
    "letter, bill, form, or product label written in Serbian, together with "
    "their question about it.\n\n"
    "Answer the question they actually asked, and only that. Do not impose a "
    "fixed structure on your reply: if they ask how a total was worked out, "
    "walk through the calculation; if they ask for a translation, translate; "
    "if they ask about one field, answer about that field. Summarise the "
    "whole document only when the question is itself a general one, and keep "
    "that summary short. Do not add advice on what to do next unless they "
    "asked for it.\n\n"
    "Use the transcript and nothing else. Quote figures, dates, and "
    "reference numbers exactly as they appear in it, and never rename a line "
    "to something more familiar: a charge you cannot identify stays "
    "unidentified. Where the transcript carries [?], that part of the page "
    "could not be read, so say so rather than filling the gap.\n\n"
    "You may do arithmetic on the figures in the transcript, and where it "
    "carries a document's own formulas, follow them. Before using a number, "
    "name the row or field it comes from, so that a misreading is visible "
    "instead of hidden inside a result. Beware of tables that number their "
    "own columns: such a number labels the column, it is never a quantity. "
    "Omitting something beats stating it wrongly.\n\n"
    "Reply in the language the user asked their question in, even though the "
    "document is in a different one: a Russian question gets a Russian "
    "answer, quoting the Serbian wording where the exact term matters.\n\n"
    "Write plain text. Telegram prints markdown as raw characters, so use no "
    "#, *, or _ for formatting; separate points with line breaks, and use "
    "plain numbers or dashes if a list helps.\n\n"
    "If the transcript shows no document or product at all, say so in one "
    "sentence. You are not a lawyer: do not give legal advice, and send the "
    "user to the issuing office for binding answers.\n\n"
    "Transcript:\n{transcript}"
)


# The [?] rule is the point: a gap is data, a guess is noise. This text is
# both what the reply is written from and what the chat remembers, so an
# invented value here would be believed for the rest of the conversation.
TRANSCRIPTION_PROMPT = (
    "Transcribe this image for a reader who cannot see it. Work top to "
    "bottom and keep the document's own structure, copying every label, "
    "number, unit and formula exactly as printed, including thousands "
    "separators and decimal commas. Do not translate, summarise, correct or "
    "explain anything, and do not skip a table because it is dense. Where a "
    "character is not legible, write [?] in its place instead of guessing: "
    "this transcript measures how much of the image you can actually read, so "
    "a gap is useful and a plausible guess is worse than nothing. Never repeat "
    "a line you have already written. If a row or a whole table is illegible, "
    "write one [?] line for it and move on to the next section rather than "
    "inventing rows that continue a pattern."
)


class TranscriptRequest(NamedTuple):
    """One transcribed photo to answer a question about.

    Attributes:
        transcript: The verbatim reading produced by the transcription chain.
        question: What the user asked, or DEFAULT_QUESTION if they sent the
            photo without a caption.
        preferences: The chat's standing rules, rendered by the orchestrator's
            _preferences_directive().
    """

    transcript: str
    question: str
    preferences: str


def encode_image_as_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Wrap raw image bytes in the data URL a vision model expects.

    The model is reached over HTTP, which carries text, so the bytes travel
    base64-encoded inside the prompt itself rather than as a file upload.

    Args:
        image_bytes: The downloaded image.
        mime_type: The image's media type, e.g. "image/jpeg".

    Returns:
        A "data:<mime>;base64,<payload>" URL.
    """
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _llm(model: str, max_tokens: int | None = None) -> ChatOpenAI:
    """Build one of the two models this agent calls.

    Args:
        model: Model to call. The two chains run on different ones, so there
            is no sensible default.
        max_tokens: Hard ceiling on the reply, or None to leave it to the API.
    """
    return ChatOpenAI(
        model=model,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
        max_tokens=max_tokens,
    )


def _image_block(detail: str) -> dict:
    """Build the templated image content block, at the given detail level."""
    return {
        "type": "image_url",
        "image_url": {"url": "{image_url}", "detail": detail},
    }


def build_summary_chain(model: str = SUMMARY_MODEL):
    """Build a chain that answers a question from a transcribed photo.

    The reply is written from the transcript rather than from the image, so
    that what the user is told and what the chat remembers come from one
    source. Two independent readings of the same photo disagree in places -
    the transcript reads a street name one way, a second look at the image
    reads it another - and since the transcript is what is stored, the bot
    would contradict itself one turn later.

    Args:
        model: Text model to call. No vision is needed here.

    Returns:
        A Runnable accepting {"transcript": ..., "question": ...,
        "preferences": ...} that returns the answer as a string.
    """
    llm = _llm(model)

    # Preferences sit right before the user's turn, the same "reminder"
    # position the translation agent needs: a standing rule placed at the top
    # of the prompt gets ignored far more often.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_PROMPT),
            ("system", "{preferences}"),
            ("human", "{question}"),
        ]
    ).partial(preferences="(no special preferences)")

    return prompt | llm | StrOutputParser()


def build_transcription_chain(detail: str = IMAGE_DETAIL, model: str = LLM_MODEL):
    """Build a chain that transcribes an image verbatim.

    The only step that looks at the image. Keeping it apart from the answering
    step is what makes a wrong reply diagnosable: the transcript shows whether
    the model misread the page or misused what it read.

    Args:
        detail: How much of the image the API should look at. The default is
            what the bot runs; scripts/probe_vision.py and the evals vary it
            to score a pairing against a bill whose values are known.
        model: Vision model to call. Overridable for the same reason.

    Returns:
        A Runnable accepting {"image_url": ...} that returns the transcript.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TRANSCRIPTION_PROMPT),
            ("human", [_image_block(detail)]),
        ]
    )

    llm = _llm(model, max_tokens=TRANSCRIPTION_MAX_TOKENS)
    return prompt | llm | StrOutputParser()


async def transcribe(chain, image_url: str) -> str:
    """Return the model's verbatim reading of one image.

    Args:
        chain: Chain built by build_transcription_chain().
        image_url: The image as a data URL.

    Returns:
        The transcript, with [?] where the model could not read a character.
    """
    return await chain.ainvoke({"image_url": image_url})


async def summarise(chain, request: TranscriptRequest) -> str:
    """Answer one question about a transcribed photo.

    Args:
        chain: Chain built by build_summary_chain().
        request: The transcript, the user's question, and their standing
            rules.

    Returns:
        The answer to send back.
    """
    return await chain.ainvoke(
        {
            "transcript": request.transcript,
            "question": request.question,
            "preferences": request.preferences,
        }
    )
