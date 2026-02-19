import logging

from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import Update
from telegram.ext import ContextTypes

from src.agents.orchestrator import process_message
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
        "/help - Show this help message\n\n"
        "You can also just type a question and I'll try to help!\n\n"
        "Examples:\n"
        '- "How do I apply for a temporary residence permit?"\n'
        '- "Translate: Dobro jutro, kako ste?"'
    )
    await update.message.reply_text(help_text)


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
        answer = await process_message(orchestrator, user_text)
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
