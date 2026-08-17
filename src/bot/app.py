import logging
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
# httpx logs every request URL at INFO, and Telegram puts the bot token in
# the URL path - so INFO here would copy the token into the deployment logs
# on every single call. WARNING keeps the failures and drops the tokens.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _open_checkpointer() -> AsyncSqliteSaver:
    """Point the orchestrator's memory at a SQLite file instead of RAM.

    Conversation history and /pref rules live in the checkpointer, so with the
    in-memory saver every redeploy silently erased them. SQLite is a plain
    file, not a database server, so this stays within the project's "no
    Postgres, no Redis" rule; put the file on a mounted volume and the state
    outlives the container.

    The connection is not awaited here: the saver connects lazily on first
    use. The constructor is not that forgiving though - it calls
    asyncio.get_running_loop(), so this may only be called once the loop is
    running, which is why the agents are built in a post_init hook.

    Returns:
        A checkpointer writing to settings.checkpoint_path.
    """
    Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpointer using %s", settings.checkpoint_path)
    return AsyncSqliteSaver(aiosqlite.connect(settings.checkpoint_path))


def _load_retriever():
    """Return a retriever, embedding the knowledge base first if it is empty."""
    vectorstore = load_vectorstore()
    if vectorstore._collection.count() == 0:
        logger.info("Vector store is empty — building from knowledge base...")
        documents = load_documents("data/knowledge_base")
        chunks = chunk_documents(documents)
        vectorstore = build_vectorstore(chunks)
        logger.info("Vector store built with %d chunks.", len(chunks))
    return get_retriever(vectorstore)


async def _initialize_agents(app) -> None:
    """Build the agents and put them in bot_data, once the loop is running.

    Registered as python-telegram-bot's `post_init` hook, which runs after
    `initialize()` and before the first update is fetched. It has to be here
    rather than in create_application() because `AsyncSqliteSaver` captures
    the running event loop in its constructor, and at the time
    create_application() is called no loop exists yet.

    Args:
        app: The Application being started, whose bot_data receives the agents.
    """
    orchestrator = build_orchestrator(
        build_rag_chain(_load_retriever()),
        build_translation_chain(),
        _open_checkpointer(),
    )
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["preference_tidier"] = build_preference_tidier()
    logger.info("Orchestrator initialized with RAG and translation agents.")


def create_application():
    """Build the bot application and register its handlers.

    Deliberately free of anything needing a running event loop: `main()` calls
    this before python-telegram-bot starts one. The agents are built later by
    the `_initialize_agents` post_init hook.

    Returns:
        Configured Application ready to be started.
    """
    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_initialize_agents)
        .build()
    )
    app.bot_data["rate_limiter"] = RateLimiter()

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
