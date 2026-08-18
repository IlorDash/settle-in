import logging
import re
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from src.agents.multimodal_agent import (
    DEFAULT_QUESTION,
    DocumentRequest,
    analyze_document,
    encode_image_as_data_url,
)
from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
    Exchange,
    add_preference,
    clear_preferences,
    get_preferences,
    preferences_directive,
    process_message,
    record_exchange,
    remove_preference,
    tidy_preferences,
)
from src.bot.feedback import VERDICT_DOWN, VERDICT_UP, record_feedback
from src.bot.middleware import (
    ValidationError,
    is_large_upload,
    validate_image_upload,
    validate_message_text,
)

logger = logging.getLogger(__name__)

ERROR_TIMEOUT = (
    "The AI service is taking too long to respond. Please try again in a moment."
)
ERROR_CONNECTION = (
    "I can't reach the AI service right now. Please try again in a few minutes."
)
ERROR_RATE_LIMIT = (
    "The AI service is currently overloaded. Please try again in a few minutes."
)
ERROR_GENERIC = (
    "Sorry, something went wrong while processing your question. "
    "Please try again later."
)

# Prefix that marks a button as ours, so the handler ignores any other
# inline keyboard added later. Full payload: "fb:<verdict>:<intent>".
CALLBACK_PREFIX = "fb"
# Only real answers are worth rating; a thumbs-down on an out-of-scope
# rejection or an error message teaches the classifier nothing.
FEEDBACK_INTENTS = frozenset({INTENT_KNOWLEDGE_QUESTION, INTENT_TRANSLATION})
FEEDBACK_THANKS = "Thanks!"
FEEDBACK_LOST = "Sorry, I can no longer find the question this answered."

# Telegram compresses every photo to JPEG, so a PhotoSize needs no mime check.
PHOTO_MIME_TYPE = "image/jpeg"
# The image itself is never stored, so the chat log keeps a note of it instead.
DOCUMENT_HISTORY_NOTE = "[photo of a document]"

# Prefix marking the "read this large file?" buttons, kept apart from the
# feedback prefix so each callback handler only ever sees its own payloads.
DOCUMENT_PREFIX = "doc"
DOCUMENT_READ = "read"
DOCUMENT_CANCEL = "cancel"

LARGE_FILE_QUESTION = (
    "That is a large file ({megabytes:.1f} MB). Sent as a file it is read at "
    "full resolution, which is worth it when the small print matters but uses "
    "several times more of the bot's capacity than an ordinary photo.\n\n"
    "Read it?"
)
LARGE_FILE_CANCELLED = (
    "Not reading it. Send the same page as an ordinary Telegram photo if you "
    "would rather keep it quick."
)
LARGE_FILE_LOST = "Sorry, I can no longer find the file this was about."
UNSUPPORTED_FILE = (
    "I can't read {kind} files yet. Please send a photo or a screenshot "
    "of the page instead."
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command. Sends a welcome message with bot capabilities."""
    welcome_text = (
        "Welcome to the Immigrant Assistance Bot!\n\n"
        "I can help you with:\n"
        "- Information about living in Serbia (residency, documents, etc.)\n"
        "- Translation between Serbian and English\n"
        "- Reading a Serbian document - just send me a photo of it\n\n"
        "Just send me a message with your question, or use /help "
        "to see available commands."
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command. Lists available commands and usage examples."""
    help_text = (
        "Available commands:\n\n"
        "/start - Welcome message\n"
        "/help - Show this help message\n"
        "/pref - Set standing preferences (e.g. always reply in Cyrillic)\n\n"
        "You can also just type a question and I'll try to help, or send a "
        "photo of a Serbian document and I'll explain it.\n\n"
        "Examples:\n"
        '- "How do I apply for a temporary residence permit?"\n'
        '- "Translate: Dobro jutro, kako ste?"\n'
        "- A photo of a bill, with or without a caption asking about it"
    )
    await update.message.reply_text(help_text)


PREF_USAGE = (
    "Preferences are standing rules I apply to your future replies "
    "(mainly how to translate).\n\n"
    "/pref - show your saved preferences\n"
    "/pref add <rule> - save a rule, e.g.\n"
    "    /pref add Write Serbian translations in Cyrillic\n"
    "/pref remove <number> - delete the rule with that number\n"
    "/pref tidy - merge rules that mean the same thing\n"
    "/pref clear - remove all your preferences"
)


def _numbered_list(rules: list[str]) -> str:
    """Render preference rules as a 1-based numbered block for /pref remove."""
    return "\n".join(f"{position}. {rule}" for position, rule in enumerate(rules, 1))


async def _pref_add(update: Update, orchestrator, chat_id, args: list[str]) -> None:
    """Save a new standing rule from `/pref add <rule>`."""
    rule = " ".join(args[1:]).strip()
    if not rule:
        await update.message.reply_text(
            "Tell me the rule to save, e.g.\n"
            "/pref add Write Serbian translations in Cyrillic"
        )
        return
    rules = await add_preference(orchestrator, chat_id, rule)
    saved = "Saved. Your preferences:\n" + _numbered_list(rules)
    await update.message.reply_text(saved)


async def _pref_remove(update: Update, orchestrator, chat_id, args: list[str]) -> None:
    """Delete one rule by its 1-based number from `/pref remove <number>`."""
    rules = await get_preferences(orchestrator, chat_id)
    if len(args) < 2 or not args[1].isdigit():
        await update.message.reply_text("Which one? Use /pref remove <number>.")
        return
    number = int(args[1])
    if number < 1 or number > len(rules):
        await update.message.reply_text(f"There is no preference number {number}.")
        return
    remaining = await remove_preference(orchestrator, chat_id, number - 1)
    if not remaining:
        await update.message.reply_text("Removed. You have no saved preferences now.")
        return
    await update.message.reply_text(
        "Removed. Your preferences:\n" + _numbered_list(remaining)
    )


async def _pref_tidy(update: Update, context, orchestrator, chat_id) -> None:
    """Merge semantically-duplicate rules on demand for `/pref tidy`."""
    tidier = context.bot_data["preference_tidier"]
    rules = await tidy_preferences(orchestrator, tidier, chat_id)
    if not rules:
        await update.message.reply_text("You have no saved preferences to tidy.")
        return
    tidied = "Tidied. Your preferences:\n" + _numbered_list(rules)
    await update.message.reply_text(tidied)


async def pref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /pref command: view, add, remove, tidy, or clear preferences.

    With no arguments it shows the saved rules. Subcommands manage them:
    `add <rule>`, `remove <number>`, `tidy` (merge duplicates), and `clear`.
    Preferences are kept per chat in the orchestrator's checkpointer, so each
    user has their own.
    """
    orchestrator = context.bot_data["orchestrator"]
    chat_id = update.message.chat_id
    args = context.args

    if not args:
        rules = await get_preferences(orchestrator, chat_id)
        if not rules:
            await update.message.reply_text(
                "You have no saved preferences yet.\n\n" + PREF_USAGE
            )
            return
        await update.message.reply_text(
            "Your saved preferences:\n" + _numbered_list(rules)
        )
        return

    action = args[0].lower()
    if action == "add":
        await _pref_add(update, orchestrator, chat_id, args)
        return
    if action == "remove":
        await _pref_remove(update, orchestrator, chat_id, args)
        return
    if action == "tidy":
        await _pref_tidy(update, context, orchestrator, chat_id)
        return
    if action == "clear":
        await clear_preferences(orchestrator, chat_id)
        await update.message.reply_text("Cleared. You have no saved preferences now.")
        return

    await update.message.reply_text(PREF_USAGE)


def _feedback_keyboard(intent: str) -> InlineKeyboardMarkup | None:
    """Build the thumbs up/down keyboard for an answer worth rating.

    The intent travels inside the button's callback data, so the tap can be
    recorded later without storing anything server-side.

    Args:
        intent: The intent the orchestrator settled on for this reply.

    Returns:
        A two-button keyboard, or None for intents we do not collect
        feedback on (out-of-scope rejections).
    """
    if intent not in FEEDBACK_INTENTS:
        return None
    buttons = [
        InlineKeyboardButton(
            "👍", callback_data=f"{CALLBACK_PREFIX}:{VERDICT_UP}:{intent}"
        ),
        InlineKeyboardButton(
            "👎", callback_data=f"{CALLBACK_PREFIX}:{VERDICT_DOWN}:{intent}"
        ),
    ]
    return InlineKeyboardMarkup([buttons])


# Markdown the models keep emitting despite being told not to. Telegram is
# sent plain text (no parse_mode), so these arrive as literal characters.
_MARKDOWN_HEADING = re.compile(r"^#{1,6}[ \t]+", flags=re.MULTILINE)
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1", flags=re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Remove the markdown Telegram would show as raw characters.

    Replies are sent without a parse_mode on purpose: MarkdownV2 demands that
    `.`, `-` and `(` be escaped, and an answer quoting "2.066,98" or a phone
    number would be rejected outright, leaving the user with an error instead
    of an answer. Cleaning the text cannot fail that way.

    Only headings and paired bold markers are touched. A lone `*` or `_` is
    left alone, since it is likelier to be part of the document being quoted
    than a formatting attempt.

    Args:
        text: The model's answer.

    Returns:
        The same answer with `#` headings and `**bold**` markers removed.
    """
    without_headings = _MARKDOWN_HEADING.sub("", text)
    return _MARKDOWN_EMPHASIS.sub(r"\2", without_headings)


async def _reply_with_error(message, error: Exception, request: str) -> None:
    """Log a failed AI call and send the user the matching apology.

    Shared by the text and the document handler so that a timeout looks the
    same to the user whichever one they hit. The user never sees a stack
    trace; the detail stays in the server log.

    Args:
        message: The user's message, to reply to.
        error: The exception the AI call raised.
        request: Short description of what was being processed, for the log.
    """
    if isinstance(error, APITimeoutError):
        logger.warning("LLM timeout for %s", request)
        await message.reply_text(ERROR_TIMEOUT)
        return
    if isinstance(error, APIConnectionError):
        logger.warning("LLM connection error for %s", request)
        await message.reply_text(ERROR_CONNECTION)
        return
    if isinstance(error, RateLimitError):
        logger.warning("OpenAI rate limit hit for %s", request)
        await message.reply_text(ERROR_RATE_LIMIT)
        return
    logger.error("Unexpected error for %s", request, exc_info=error)
    await message.reply_text(ERROR_GENERIC)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any non-command text message.

    Validates input, checks rate limit, then routes through the orchestrator.
    """
    try:
        user_text = validate_message_text(update.message.text)
        rate_limiter = context.bot_data["rate_limiter"]
        rate_limiter.check(update.message.from_user.id)
    except ValidationError as e:
        await update.message.reply_text(e.user_message)
        return

    orchestrator = context.bot_data["orchestrator"]

    user_id = update.message.from_user.id

    try:
        result = await process_message(
            orchestrator, user_text, thread_id=update.message.chat_id
        )
        # Answering as a Telegram reply is load-bearing, not cosmetic: it is
        # how a later button tap finds the question this answered.
        await update.message.reply_text(
            _strip_markdown(result.response),
            reply_to_message_id=update.message.message_id,
            # If the question was deleted while the LLM was thinking, send
            # the answer anyway rather than losing a reply we paid for.
            allow_sending_without_reply=True,
            reply_markup=_feedback_keyboard(result.intent),
        )
    except Exception as error:
        await _reply_with_error(update.message, error, f"user {user_id}: {user_text}")


def _uploaded_image(message) -> tuple:
    """Pick the image out of a photo message or an image sent as a file.

    Args:
        message: The incoming Telegram message.

    Returns:
        The file to download and its media type.
    """
    if message.photo:
        # Telegram attaches the same photo at several sizes; the last is the
        # largest, and Telegram has already re-encoded all of them as JPEG.
        return message.photo[-1], PHOTO_MIME_TYPE
    return message.document, message.document.mime_type


async def _fetch_document_image(message) -> str:
    """Validate an uploaded image and download it as a data URL.

    Size and type come from the update itself, so an oversized file is
    refused before a byte of it is fetched.

    Args:
        message: The incoming Telegram message.

    Returns:
        The image encoded as a data URL, ready for the vision agent.

    Raises:
        ValidationError: If the image is too large or of an unusable type.
    """
    upload, mime_type = _uploaded_image(message)
    validate_image_upload(upload.file_size, mime_type)
    # Logged because the model is billed by the pixels it is handed, and
    # "what did it actually see" was guesswork until this line existed.
    logger.info(
        "Reading a %s upload of %s bytes (%s)",
        "photo" if message.photo else "file",
        upload.file_size,
        mime_type,
    )
    telegram_file = await upload.get_file()
    image_bytes = await telegram_file.download_as_bytearray()
    return encode_image_as_data_url(bytes(image_bytes), mime_type)


def _document_history_note(caption: str | None) -> str:
    """Describe a photo turn in words, since the image itself is not stored."""
    if caption:
        return f"{DOCUMENT_HISTORY_NOTE} {caption}"
    return DOCUMENT_HISTORY_NOTE


def _confirmation_keyboard() -> InlineKeyboardMarkup:
    """Build the yes/no buttons offered before reading a large file."""
    buttons = [
        InlineKeyboardButton(
            "Read it", callback_data=f"{DOCUMENT_PREFIX}:{DOCUMENT_READ}"
        ),
        InlineKeyboardButton(
            "Cancel", callback_data=f"{DOCUMENT_PREFIX}:{DOCUMENT_CANCEL}"
        ),
    ]
    return InlineKeyboardMarkup([buttons])


async def _read_photo_and_reply(message, context) -> None:
    """Send the photo on `message` to the vision agent and answer it.

    Shared by the direct path and the confirmed-large-file path, so both cost
    the same checks and record the same history.

    Args:
        message: The Telegram message carrying the photo or image file.
        context: The PTB context, holding the agents in bot_data.
    """
    orchestrator = context.bot_data["orchestrator"]
    chat_id = message.chat_id
    context.bot_data["rate_limiter"].check(message.from_user.id)
    request = DocumentRequest(
        image_url=await _fetch_document_image(message),
        question=message.caption or DEFAULT_QUESTION,
        preferences=await preferences_directive(orchestrator, chat_id),
    )
    answer = await analyze_document(context.bot_data["multimodal_chain"], request)
    await message.reply_text(_strip_markdown(answer))
    await record_exchange(
        orchestrator,
        chat_id,
        Exchange(_document_history_note(message.caption), answer),
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Read any photo the user sends and answer their question about it.

    An image carries no text to classify, so the modality alone chooses the
    agent: this skips the orchestrator's graph and calls the vision agent
    directly. The turn is written into the chat's history afterwards, so a
    text follow-up about the document still has the context.

    A large file is not read straight away. Telegram compresses anything sent
    as a photo, so only an uncompressed upload can be big enough to matter,
    and that one is offered as a choice first.
    """
    message = update.message
    user_id = message.from_user.id
    try:
        upload, mime_type = _uploaded_image(message)
        # Checked before the offer: agreeing to read a file we would then
        # refuse wastes the user's time and makes the question look pointless.
        validate_image_upload(upload.file_size, mime_type)
        if message.document and is_large_upload(upload.file_size):
            await message.reply_text(
                LARGE_FILE_QUESTION.format(megabytes=upload.file_size / (1024 * 1024)),
                reply_to_message_id=message.message_id,
                allow_sending_without_reply=True,
                reply_markup=_confirmation_keyboard(),
            )
            return
        await _read_photo_and_reply(message, context)
    except ValidationError as e:
        await message.reply_text(e.user_message)
    except Exception as error:
        # Downloading can fail too, not just the model call, so the net is
        # cast around the whole thing rather than the vision call alone.
        await _reply_with_error(message, error, f"user {user_id}: a document photo")


async def document_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Read a large file, or drop it, according to which button was tapped.

    Stateless in the same way as the feedback buttons: nothing is held
    between the question and the answer, because the confirmation was sent as
    a reply to the upload, so the file is found again through
    `query.message.reply_to_message`.
    """
    query = update.callback_query
    # Clearing the buttons doubles as a lock. Telegram refuses the edit when
    # they are already gone, which means another tap got here first, and
    # reading again would pay for the same file twice.
    if not await _hide_keyboard(query):
        await query.answer()
        return

    if query.data == f"{DOCUMENT_PREFIX}:{DOCUMENT_CANCEL}":
        # Answered as a toast rather than a message: the upload may be too
        # old for Telegram to let us reply to it, and there is nothing to say
        # that is worth a second failure path.
        await query.answer(LARGE_FILE_CANCELLED)
        return

    upload = _replied_message(query)
    if upload is None:
        await query.answer(LARGE_FILE_LOST)
        return

    await query.answer()
    try:
        await _read_photo_and_reply(upload, context)
    except ValidationError as e:
        await upload.reply_text(e.user_message)
    except Exception as error:
        user_id = upload.from_user.id
        await _reply_with_error(upload, error, f"user {user_id}: a confirmed file")


def _file_kind(document) -> str:
    """Name the file type in words the sender will recognise.

    Args:
        document: The Telegram Document that arrived.

    Returns:
        An extension such as "DOCX", or a fallback when there is no name.
    """
    name = document.file_name or ""
    suffix = Path(name).suffix.lstrip(".")
    return suffix.upper() if suffix else "that kind of"


async def handle_unsupported_file(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Say which file type arrived and cannot be read, rather than going quiet.

    Registered after the image handler, so it only sees what that one turned
    down: PDFs, documents, archives. A vision model reads images, and
    rendering anything else to one needs tools the container does not carry.
    """
    await update.message.reply_text(
        UNSUPPORTED_FILE.format(kind=_file_kind(update.message.document))
    )


def _parse_feedback(callback_data: str) -> tuple[str, str] | None:
    """Split a button payload into its verdict and intent.

    Buttons outlive deploys: a keyboard sent before this format existed is
    still tappable in the chat history, so unknown payloads are rejected
    rather than unpacked.

    Args:
        callback_data: The raw payload from the tapped button.

    Returns:
        A (verdict, intent) pair, or None if the payload is not ours.
    """
    parts = callback_data.split(":", maxsplit=2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    _, verdict, intent = parts
    # Both fields are checked against the values we issue, so the stored
    # dataset keeps a closed set of labels to retrain the classifier on.
    if verdict not in (VERDICT_UP, VERDICT_DOWN):
        return None
    if intent not in FEEDBACK_INTENTS:
        return None
    return verdict, intent


def _replied_message(query: CallbackQuery):
    """Recover the message a tapped button's message was replying to.

    Both button features are stateless and rely on this one link: the bot
    answers as a Telegram reply, so a later tap can walk back to whatever
    prompted it without anything being stored in between.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        The message that was replied to, or None if the chain back to it is
        broken (the user deleted it, or the message is too old for Telegram
        to still hand us its contents).
    """
    tapped = query.message
    # Past about 48 hours Telegram sends an InaccessibleMessage, which
    # carries only a chat and an id - reading anything else raises.
    if tapped is None or not tapped.is_accessible:
        return None
    return tapped.reply_to_message


def _rated_question(query: CallbackQuery) -> str | None:
    """Recover the text of the question that the rated answer replied to.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        The original question text, or None if it can no longer be reached.
    """
    question = _replied_message(query)
    if question is None:
        return None
    return question.text or None


async def _hide_keyboard(query: CallbackQuery) -> bool:
    """Remove the buttons after a tap, tolerating a message that cannot change.

    Shared by both button features: once a choice is made, its buttons go.
    The result also says whether this call was the one that removed them,
    which is how a second tap on the same keyboard is recognised.

    Telegram rejects the edit when the keyboard is already gone, and
    python-telegram-bot raises outright when the message is inaccessible.
    The vote is stored by this point, so neither may reach the user.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        True if this call removed the buttons, False if they were already
        gone or the message can no longer be edited.
    """
    try:
        await query.edit_message_reply_markup(None)
    except (BadRequest, TypeError):
        logger.debug("Keyboard could not be cleared (stale message).")
        return False
    return True


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record a thumbs up/down tap on an answer, then clear the buttons.

    Holds no state between the answer and the tap: the question is read
    back from the message the answer replied to.
    """
    query = update.callback_query
    parsed = _parse_feedback(query.data)
    if parsed is None:
        await query.answer()
        return

    question = _rated_question(query)
    if question is None:
        await query.answer(FEEDBACK_LOST)
        return

    verdict, intent = parsed
    try:
        record_feedback(question, intent, verdict)
    except OSError:
        # Feedback is training data, not the user's business: a full or
        # read-only disk must not turn their tap into an error.
        logger.exception("Could not save feedback for intent %s", intent)

    await query.answer(FEEDBACK_THANKS)
    await _hide_keyboard(query)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log an exception no handler caught, and apologise where possible.

    Registered with `add_error_handler`, so it is the last line of defence:
    without it an unexpected crash leaves the user with silence. `update` is
    typed `object` because the failure need not come from a message at all,
    in which case there is nobody to reply to.

    Args:
        update: The update being processed when the error happened, if any.
        context: The PTB context, carrying the exception on `.error`.
    """
    logger.error("Unhandled error while processing an update", exc_info=context.error)

    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await message.reply_text(ERROR_GENERIC)
    except TelegramError:
        # The chat may be blocked or gone; the log above is what matters.
        logger.debug("Could not deliver the error notice to the user.")
