import logging

from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import Update
from telegram.ext import ContextTypes

from src.agents.orchestrator import (
    add_preference,
    clear_preferences,
    get_preferences,
    process_message,
    remove_preference,
    tidy_preferences,
)
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
        answer = await process_message(
            orchestrator, user_text, thread_id=update.message.chat_id
        )
        await update.message.reply_text(answer)
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
