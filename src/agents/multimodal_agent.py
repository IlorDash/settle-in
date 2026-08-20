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

# Sends the image at its own resolution, budgeted in 32x32 patches up to
# roughly 10 megapixels, instead of the 768px shortest side gpt-4o-class
# models impose. Only gpt-5.4 and newer accept it, so this and LLM_MODEL move
# together; tests/evals/test_vision_evals.py scores any pairing.
IMAGE_DETAIL = "original"

# A dense bill transcribes to roughly 2000 tokens. The cap is headroom: a
# model that cannot read a table repeats one line until something stops it.
TRANSCRIPTION_MAX_TOKENS = 4000

# What to ask the model when the user sent a photo with no caption. The chat
# remembers what the bot said, not what it read, so anything left out of this
# first answer cannot be asked about later: a summary that skipped a table of
# monthly figures made a follow-up about them unanswerable.
DEFAULT_QUESTION = (
    "What is this, and what does it say? Work through it section by section "
    "and include the figures each one holds, because I may ask about any of "
    "them next."
)

SYSTEM_PROMPT = (
    "You are an assistant for immigrants in Serbia. The user sends a photo, "
    "usually an official document, letter, bill, form, or product label "
    "written in Serbian, together with a question about it.\n\n"
    "Answer the question they actually asked, and only that. Do not impose a "
    "fixed structure on your reply: if they ask how a total was worked out, "
    "walk through the calculation; if they ask for a translation, translate; "
    "if they ask about one field, answer about that field. Summarise the "
    "whole document only when the question is itself a general one. Do not "
    "add advice on what to do next unless they asked for it.\n\n"
    "Read the image carefully before answering. Quote figures, dates, and "
    "reference numbers exactly as printed, and say plainly which parts you "
    "cannot read rather than filling the gap with something plausible. Never "
    "invent a value, and never rename a line to something more familiar: a "
    "charge you cannot identify stays unidentified.\n\n"
    "You may do arithmetic on numbers you can actually see, and where a "
    "document prints its own formulas, follow them. Before using a number, "
    "name the row or field it comes from and quote it as printed, so that a "
    "misreading is visible instead of hidden inside a result. Beware of "
    "tables that number their own columns: such a number labels the column, "
    "it is never a quantity. If you cannot read a value with confidence, say "
    "so rather than calculating with a guess. Omitting something beats "
    "stating it wrongly.\n\n"
    "Reply in the language the user asked their question in, even though the "
    "document is in a different one: a Russian question gets a Russian "
    "answer, quoting the Serbian wording where the exact term matters.\n\n"
    "Write plain text. Telegram prints markdown as raw characters, so use no "
    "#, *, or _ for formatting; separate points with line breaks, and use "
    "plain numbers or dashes if a list helps.\n\n"
    "If the photo shows no document or product at all, say so in one "
    "sentence. You are not a lawyer: do not give legal advice, and send the "
    "user to the issuing office for binding answers."
)


# Used only by scripts/probe_vision.py and the evals, to measure reading
# rather than reasoning. The [?] rule is the point: a gap is data, a guess is
# noise.
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


class DocumentRequest(NamedTuple):
    """One document for the vision agent to read.

    Attributes:
        image_url: The image as a data URL, from encode_image_as_data_url().
        question: What the user asked, or DEFAULT_QUESTION if they sent the
            photo without a caption.
        preferences: The chat's standing rules, rendered by the orchestrator's
            _preferences_directive().
    """

    image_url: str
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


def _vision_llm(model: str = LLM_MODEL, max_tokens: int | None = None) -> ChatOpenAI:
    """Build the vision model both the answering and transcribing chains use.

    Args:
        model: Vision model to call.
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


def build_multimodal_chain(detail: str = IMAGE_DETAIL, model: str = LLM_MODEL):
    """Build an LCEL chain that reads a photo and answers a question about it.

    Args:
        detail: How much of the image the API should look at. The default is
            what the bot runs; `scripts/probe_vision.py` varies it to measure
            what the model can actually read.
        model: Vision model to call. Overridable for the same reason.

    Returns:
        A Runnable accepting {"image_url": ..., "question": ...,
        "preferences": ...} that returns the answer as a string.
    """
    llm = _vision_llm(model)

    # The human turn is a list of content blocks, not a string: the text and
    # the image are two parts of one message. Preferences sit right before it
    # for the same reason as in the translation agent - a standing rule placed
    # at the top of the prompt gets ignored far more often.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("system", "{preferences}"),
            (
                "human",
                [
                    {"type": "text", "text": "{question}"},
                    _image_block(detail),
                ],
            ),
        ]
    ).partial(preferences="(no special preferences)")

    return prompt | llm | StrOutputParser()


def build_transcription_chain(detail: str = IMAGE_DETAIL, model: str = LLM_MODEL):
    """Build a chain that transcribes an image verbatim, for diagnosis.

    Not part of the bot's reply path. The answering chain reads and reasons in
    one pass, so when an answer is wrong there is no intermediate text to
    inspect and a misreading cannot be told apart from a reasoning mistake.
    This produces that text deliberately, which makes "how much of the image
    can the model resolve" measurable, and comparable across detail settings.

    Args:
        detail: How much of the image the API should look at.
        model: Vision model to call.

    Returns:
        A Runnable accepting {"image_url": ...} that returns the transcript.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TRANSCRIPTION_PROMPT),
            ("human", [_image_block(detail)]),
        ]
    )

    llm = _vision_llm(model, max_tokens=TRANSCRIPTION_MAX_TOKENS)
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


async def analyze_document(chain, request: DocumentRequest) -> str:
    """Send one document photo through the vision chain.

    Args:
        chain: Chain built by build_multimodal_chain().
        request: The image, the user's question, and their standing rules.

    Returns:
        The model's explanation of the document.
    """
    return await chain.ainvoke(
        {
            "image_url": request.image_url,
            "question": request.question,
            "preferences": request.preferences,
        }
    )
