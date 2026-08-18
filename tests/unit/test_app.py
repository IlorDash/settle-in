import logging
from unittest.mock import AsyncMock, MagicMock, patch

import src.bot.app  # noqa: F401  (imported for its logging configuration)
from src.bot.app import (
    _close_checkpointer,
    _initialize_agents,
    create_application,
)


def test_httpx_logging_is_pinned_to_warning_to_keep_the_token_out_of_logs():
    # Telegram puts the bot token in the request URL, and httpx logs every
    # URL at INFO. Left on, that copies the token into the deployment logs
    # on every call, so anyone with log access owns the bot.
    #
    # This asserts the logger's own level rather than isEnabledFor(): pytest
    # attaches its own root handlers, which makes logging.basicConfig a no-op
    # and leaves the root logger at WARNING. Every logger would then look
    # "not enabled for INFO" during tests, so isEnabledFor would pass just as
    # happily with this protection removed.
    assert logging.getLogger("httpx").level == logging.WARNING


def test_create_application_needs_no_running_event_loop():
    # main() calls this before python-telegram-bot starts the loop, so nothing
    # here may require one. AsyncSqliteSaver captures the running loop in its
    # constructor, and building it here crashed the bot on startup with
    # "RuntimeError: no running event loop" while every test stayed green.
    #
    # Deliberately NOT an async test: pytest-asyncio would supply the very
    # loop whose absence is the bug, and this would pass unfixed.
    app = create_application()

    assert app.bot_data["rate_limiter"] is not None


def test_create_application_registers_the_agent_hook():
    # Nothing else builds the agents, so dropping the post_init registration
    # would start a bot whose bot_data has no orchestrator, and every message
    # would then fail with a KeyError.
    app = create_application()

    assert app.post_init is _initialize_agents


def test_create_application_registers_the_shutdown_hook():
    # Without this hook the aiosqlite worker thread, which is not a daemon,
    # keeps the interpreter alive and Ctrl+C leaves the bot hanging after
    # "Application.stop() complete".
    app = create_application()

    assert app.post_shutdown is _close_checkpointer


async def test_post_shutdown_closes_the_checkpointer_connection():
    # Closing the connection is what stops that worker thread; nothing else
    # in the shutdown path knows the checkpointer exists.
    connection = AsyncMock()
    app = MagicMock()
    app.bot_data = {"checkpointer": MagicMock(conn=connection)}

    await _close_checkpointer(app)

    connection.close.assert_awaited_once()


async def test_post_shutdown_survives_a_failed_startup():
    # post_init can raise before it stores anything, and post_shutdown still
    # runs; a KeyError here would mask the real startup error.
    app = MagicMock()
    app.bot_data = {}

    await _close_checkpointer(app)


@patch("src.bot.app.build_preference_tidier", MagicMock())
@patch("src.bot.app.build_translation_chain", MagicMock())
@patch("src.bot.app.build_rag_chain", MagicMock())
@patch("src.bot.app.build_orchestrator")
@patch("src.bot.app._open_checkpointer", MagicMock())
@patch("src.bot.app._load_retriever", MagicMock())
async def test_post_init_puts_the_orchestrator_in_bot_data(mock_build):
    # The agents move into bot_data only when the post_init hook runs, so the
    # handlers would find nothing there if it were left unregistered.
    app = MagicMock()
    app.bot_data = {}

    await _initialize_agents(app)

    assert app.bot_data["orchestrator"] is mock_build.return_value
