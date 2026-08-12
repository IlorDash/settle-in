import logging

from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
    add_preference,
    clear_preferences,
    get_preferences,
    process_message,
    remove_preference,
    tidy_preferences,
)
from src.bot.feedback import VERDICT_DOWN, VERDICT_UP, record_feedback
from src.bot.middleware import ValidationError, validate_message_text

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


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command. Sends a welcome message with bot capabilities."""
    welcome_text = (
        "Welcome to the Immigrant Assistance Bot!\n\n"
        "I can help you with:\n"
        "- Information about living in Serbia (residency, documents, etc.)\n"
        "- Translation between Serbian and English\n\n"
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
        "You can also just type a question and I'll try to help!\n\n"
        "Examples:\n"
        '- "How do I apply for a temporary residence permit?"\n'
        '- "Translate: Dobro jutro, kako ste?"'
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
    rules = add_preference(orchestrator, chat_id, rule)
    saved = "Saved. Your preferences:\n" + _numbered_list(rules)
    await update.message.reply_text(saved)


async def _pref_remove(update: Update, orchestrator, chat_id, args: list[str]) -> None:
    """Delete one rule by its 1-based number from `/pref remove <number>`."""
    rules = get_preferences(orchestrator, chat_id)
    if len(args) < 2 or not args[1].isdigit():
        await update.message.reply_text("Which one? Use /pref remove <number>.")
        return
    number = int(args[1])
    if number < 1 or number > len(rules):
        await update.message.reply_text(f"There is no preference number {number}.")
        return
    remaining = remove_preference(orchestrator, chat_id, number - 1)
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
        rules = get_preferences(orchestrator, chat_id)
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
        clear_preferences(orchestrator, chat_id)
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
            result.response,
            reply_to_message_id=update.message.message_id,
            # If the question was deleted while the LLM was thinking, send
            # the answer anyway rather than losing a reply we paid for.
            allow_sending_without_reply=True,
            reply_markup=_feedback_keyboard(result.intent),
        )
    except APITimeoutError:
        logger.warning("LLM timeout for user %s: %s", user_id, user_text)
        await update.message.reply_text(ERROR_TIMEOUT)
    except APIConnectionError:
        logger.warning("LLM connection error for user %s", user_id)
        await update.message.reply_text(ERROR_CONNECTION)
    except RateLimitError:
        logger.warning("OpenAI rate limit hit for user %s", user_id)
        await update.message.reply_text(ERROR_RATE_LIMIT)
    except Exception:
        logger.exception("Unexpected error for user %s: %s", user_id, user_text)
        await update.message.reply_text(ERROR_GENERIC)


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


def _rated_question(query: CallbackQuery) -> str | None:
    """Recover the question that the rated answer was replying to.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        The original question text, or None if the chain back to it is
        broken (the user deleted the question, or the answer is too old
        for Telegram to still hand us its contents).
    """
    answer = query.message
    # Past about 48 hours Telegram sends an InaccessibleMessage, which
    # carries only a chat and an id - reading anything else raises.
    if answer is None or not answer.is_accessible:
        return None
    if answer.reply_to_message is None:
        return None
    return answer.reply_to_message.text or None


async def _hide_feedback_keyboard(query: CallbackQuery) -> None:
    """Remove the buttons after a tap, tolerating a message that cannot change.

    Telegram rejects the edit when the keyboard is already gone, and
    python-telegram-bot raises outright when the message is inaccessible.
    The vote is stored by this point, so neither may reach the user.

    Args:
        query: The CallbackQuery raised by the button tap.
    """
    try:
        await query.edit_message_reply_markup(None)
    except (BadRequest, TypeError):
        logger.debug("Feedback keyboard could not be cleared (stale message).")


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
    await _hide_feedback_keyboard(query)


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
