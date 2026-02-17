import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.agents.orchestrator import process_message

logger = logging.getLogger(__name__)


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

    Routes through the LangGraph orchestrator.
    """
    user_text = update.message.text
    orchestrator = context.bot_data["orchestrator"]

    try:
        answer = await process_message(orchestrator, user_text)
        await update.message.reply_text(answer)
    except Exception:
        logger.exception("Orchestrator failed for query: %s", user_text)
        await update.message.reply_text(
            "Sorry, something went wrong while processing your question. "
            "Please try again later."
        )
