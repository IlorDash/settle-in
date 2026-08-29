import asyncio
import contextlib
import logging
from pathlib import Path

import aiosqlite
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telegram import Bot, BotCommand, BotCommandScopeChat
from telegram.error import TelegramError
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
from src.bot.admin import TelegramLogHandler, mirror_logs
from src.bot.handlers import (
    ADMIN_PREFIX,
    CALLBACK_PREFIX,
    DOCUMENT_PREFIX,
    admin_callback,
    admin_command,
    default_features,
    document_callback,
    error_handler,
    export_command,
    feedback_callback,
    handle_message,
    handle_photo,
    handle_unsupported_file,
    help_command,
    limits_command,
    loglevel_command,
    logs_command,
    pref_command,
    reset_command,
    start_command,
)
from src.bot.middleware import DailyQuota, RateLimiter
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

# The "/" menu Telegram offers in a chat. Scopes do not merge: a list set
# for one chat replaces the default list there rather than adding to it, so
# the operator's menu has to repeat the public commands or lose them.
PUBLIC_COMMANDS = (
    ("start", "Welcome message"),
    ("help", "What I can do"),
    ("pref", "Standing preferences for my replies"),
    ("reset", "Forget our conversation"),
    ("export", "Send our recent messages as a file"),
    ("limits", "Your daily allowance, and what is left of it"),
)
# /limits is public: it answers everyone, and only the setting half of it is
# the operator's. It belongs in one list or the other, never both - the
# operator's menu is these two concatenated.
OPERATOR_COMMANDS = (
    ("admin", "Operator panel"),
    ("loglevel", "Which records reach this chat"),
    ("logs", "Send the log held in memory"),
)
# Published once, at startup: the menu lists what the bot has, not what it
# will do right now, so it still offers /export and /reset while the panel
# has one of them switched off. The command itself says when it is off.


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


class StartupError(RuntimeError):
    """The bot cannot serve anyone, so it must not pretend to be up."""


def _load_retriever_or_stop() -> VectorStoreRetriever:
    """Return a retriever, or stop the bot if the knowledge base is unusable.

    Retrieval is the main feature, so a bot that starts without it would
    answer every question with an apology. Better to fail loudly: the log
    line names the cause and the process exits non-zero.

    Returns:
        The retriever, ready to search.

    Raises:
        StartupError: If the knowledge base could not be loaded or indexed.
    """
    try:
        return _load_retriever()
    except Exception as error:
        logger.critical(
            "Cannot start: the knowledge base could not be indexed.", exc_info=error
        )
        raise StartupError("knowledge base indexing failed") from error


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
    checkpointer = _open_checkpointer()
    # Kept so _close_checkpointer can shut its worker thread down on exit,
    # and stored before anything that can fail: post_shutdown runs even when
    # this hook raises, and it can only close what it can find.
    app.bot_data["checkpointer"] = checkpointer
    orchestrator = build_orchestrator(
        build_rag_chain(_load_retriever_or_stop()),
        build_translation_chain(),
        checkpointer,
    )
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["preference_tidier"] = build_preference_tidier()
    # Not persisted: a restart is the one thing that reliably puts every
    # switch back where the bot ships it.
    app.bot_data["features"] = default_features()
    _start_log_mirror(app)
    await _publish_operator_menu(app.bot)
    logger.info("Orchestrator initialized with RAG and translation agents.")


def _start_log_mirror(app) -> None:
    """Send this run's serious log records to the operator, if there is one.

    Does nothing without ADMIN_CHAT_ID, so a deployment that has not set it
    behaves exactly as before. Installed here rather than at import because
    the sending task needs a running event loop, which is the same reason
    the agents are built in this hook.

    Args:
        app: The Application being started, whose bot_data receives both.
    """
    if not settings.admin_chat_id:
        return

    handler = TelegramLogHandler()
    logging.getLogger().addHandler(handler)
    app.bot_data["log_handler"] = handler
    app.bot_data["log_mirror"] = asyncio.create_task(mirror_logs(app.bot, handler))
    logger.info("Mirroring logs to chat %s.", settings.admin_chat_id)


async def _publish_operator_menu(bot: Bot) -> None:
    """Offer the operator's commands in the "/" menu of their chat alone.

    Presentation only: it keeps the operator commands out of everyone
    else's menu, but it does not protect them - the is_admin guard in each
    of their handlers is what does that, and every command added to
    OPERATOR_COMMANDS needs one.

    A menu is worth nothing next to a bot that will not start, so a chat
    Telegram refuses to set one for - the usual cause is an ADMIN_CHAT_ID
    that has never messaged the bot - is logged and let go.

    Args:
        bot: The bot whose command menu is being set.
    """
    if not settings.admin_chat_id:
        return

    commands = [
        BotCommand(name, description)
        for name, description in PUBLIC_COMMANDS + OPERATOR_COMMANDS
    ]
    try:
        await bot.set_my_commands(
            commands, scope=BotCommandScopeChat(settings.admin_chat_id)
        )
    except TelegramError as error:
        logger.warning(
            "Could not offer the operator menu in chat %s: %s",
            settings.admin_chat_id,
            error,
        )


async def _shut_down_cleanly(app) -> None:
    """Stop everything this run started, in the order it was started.

    Registered as python-telegram-bot's `post_shutdown` hook, which runs
    even when startup itself failed - so both halves have to tolerate the
    thing they close never having been created.

    Args:
        app: The Application being shut down.
    """
    await _stop_log_mirror(app)
    await _close_checkpointer(app)


async def _stop_log_mirror(app) -> None:
    """Detach the log handler and wait for its forwarding task to stop.

    The handler is taken off the root logger first, so no record is queued
    once there is nobody left to send it - and so a second `initialize()`
    in the same process does not stack a second handler onto the first.

    The task is awaited rather than only cancelled: `cancel()` merely
    schedules the CancelledError, and this hook is the last thing awaited
    before the loop closes, so a task left unresumed prints "Task was
    destroyed but it is pending" on the way out.

    Args:
        app: The Application being shut down.
    """
    handler = app.bot_data.get("log_handler")
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        handler.close()

    mirror = app.bot_data.get("log_mirror")
    if mirror is None:
        return
    mirror.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await mirror


async def _close_checkpointer(app) -> None:
    """Close the SQLite connection so the process can actually exit.

    aiosqlite runs every query on a worker thread that is NOT a daemon and
    that blocks on a queue until the connection is closed. Nothing else
    closes it, so without this the interpreter waits on that thread forever
    and Ctrl+C leaves the bot hanging after "Application.stop() complete".

    Args:
        app: The Application being shut down.
    """
    checkpointer = app.bot_data.get("checkpointer")
    if checkpointer is None:
        return
    await checkpointer.conn.close()
    logger.info("Checkpointer connection closed.")


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
        .post_shutdown(_shut_down_cleanly)
        .build()
    )
    app.bot_data["rate_limiter"] = RateLimiter()
    # The slower half of the same job: the rate limiter caps how fast one
    # user spends, this caps how much a chat spends in a day. Built here
    # rather than in post_init because it needs no event loop.
    app.bot_data["daily_quota"] = DailyQuota(
        text_limit=settings.daily_text_limit,
        photo_limit=settings.daily_photo_limit,
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("pref", pref_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("export", export_command))
    # Operator commands. They answer only in the admin chat; anywhere else
    # they are silent, so there is nothing to discover by guessing.
    app.add_handler(CommandHandler("limits", limits_command))
    app.add_handler(CommandHandler("loglevel", loglevel_command))
    app.add_handler(CommandHandler("logs", logs_command))
    app.add_handler(CommandHandler("admin", admin_command))
    # The pattern keeps this handler off any inline keyboard added later.
    app.add_handler(
        CallbackQueryHandler(feedback_callback, pattern=f"^{CALLBACK_PREFIX}:")
    )
    app.add_handler(
        CallbackQueryHandler(document_callback, pattern=f"^{DOCUMENT_PREFIX}:")
    )
    app.add_handler(CallbackQueryHandler(admin_callback, pattern=f"^{ADMIN_PREFIX}:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Modality picks the agent here: an image never reaches intent
    # classification. Document.IMAGE covers a photo sent as an uncompressed
    # file, which is how people keep small print readable.
    app.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo)
    )
    # Registered after the image handler, so it catches only what that one
    # turned down. Without it any other file type would vanish in silence.
    app.add_handler(MessageHandler(filters.Document.ALL, handle_unsupported_file))
    # Last line of defence: catches anything the handlers above let through.
    app.add_error_handler(error_handler)

    logger.info("Bot handlers registered successfully.")
    return app


def main() -> None:
    """Start the bot, turning a refused startup into a failing exit code."""
    logger.info("Starting bot in %s mode...", settings.bot_mode)

    app = create_application()
    try:
        _run(app)
    except StartupError:
        # The cause is already logged at CRITICAL; re-raising it here would
        # bury that line under a traceback. Exiting non-zero is what tells
        # the platform the start failed rather than finished.
        raise SystemExit(1) from None


def _run(app) -> None:
    """Hand control to python-telegram-bot in the configured mode.

    Args:
        app: The built Application, ready to start.
    """
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
