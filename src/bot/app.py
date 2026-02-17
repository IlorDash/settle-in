import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from src.agents.orchestrator import build_orchestrator
from src.agents.rag_agent import build_rag_chain
from src.agents.translation_agent import build_translation_chain
from src.bot.handlers import handle_message, help_command, start_command
from src.bot.middleware import RateLimiter
from src.config import settings
from src.knowledge.vectorstore import get_retriever, load_vectorstore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_application():
    """Build the bot application, initialize the orchestrator, and register handlers.

    Loads the vector store, builds RAG and translation chains, compiles them
    into a LangGraph orchestrator, and stores it in bot_data so handlers
    can access it via context.bot_data["orchestrator"].

    Returns:
        Configured Application ready to be started.
    """
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    translation_chain = build_translation_chain()
    orchestrator = build_orchestrator(rag_chain, translation_chain)
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["rate_limiter"] = RateLimiter()
    logger.info("Orchestrator initialized with RAG and translation agents.")

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot handlers registered successfully.")
    return app


def main() -> None:
    """Start the bot in the mode specified by BOT_MODE env variable.

    Raises:
        NotImplementedError: If bot_mode is "webhook" (not yet supported).
    """
    logger.info("Starting bot in %s mode...", settings.bot_mode)

    app = create_application()

    if settings.bot_mode == "webhook":
        raise NotImplementedError("Webhook mode will be added in Phase 5.")

    logger.info("Bot is polling for updates... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
