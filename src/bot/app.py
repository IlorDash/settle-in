import logging

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.agents.orchestrator import build_orchestrator, build_preference_tidier
from src.agents.rag_agent import build_rag_chain
from src.agents.translation_agent import build_translation_chain
from src.bot.handlers import (
    CALLBACK_PREFIX,
    error_handler,
    feedback_callback,
    handle_message,
    help_command,
    pref_command,
    start_command,
)
from src.bot.middleware import RateLimiter
from src.config import settings
from src.knowledge.loader import chunk_documents, load_documents
from src.knowledge.vectorstore import build_vectorstore, get_retriever, load_vectorstore

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
    if vectorstore._collection.count() == 0:
        logger.info("Vector store is empty — building from knowledge base...")
        documents = load_documents("data/knowledge_base")
        chunks = chunk_documents(documents)
        vectorstore = build_vectorstore(chunks)
        logger.info("Vector store built with %d chunks.", len(chunks))
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    translation_chain = build_translation_chain()
    orchestrator = build_orchestrator(rag_chain, translation_chain)
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["preference_tidier"] = build_preference_tidier()
    app.bot_data["rate_limiter"] = RateLimiter()
    logger.info("Orchestrator initialized with RAG and translation agents.")

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("pref", pref_command))
    # The pattern keeps this handler off any inline keyboard added later.
    app.add_handler(
        CallbackQueryHandler(feedback_callback, pattern=f"^{CALLBACK_PREFIX}:")
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Last line of defence: catches anything the handlers above let through.
    app.add_error_handler(error_handler)

    logger.info("Bot handlers registered successfully.")
    return app


def main() -> None:
    """Start the bot in the mode specified by BOT_MODE env variable."""
    logger.info("Starting bot in %s mode...", settings.bot_mode)

    app = create_application()

    if settings.bot_mode == "webhook":
        logger.info(
            "Bot is starting webhook on 0.0.0.0:%s ... "
            "Telegram will POST updates to %s",
            settings.port,
            settings.webhook_url,
        )
        app.run_webhook(
            listen="0.0.0.0",
            port=settings.port,
            url_path="/webhook",
            webhook_url=settings.webhook_url,
        )
        return

    logger.info("Bot is polling for updates... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
