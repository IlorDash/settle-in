import asyncio
import contextlib
import logging
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import src.bot.app  # noqa: F401  (imported for its logging configuration)
from src.bot.app import (
    StartupError,
    _close_checkpointer,
    _initialize_agents,
    _shut_down_cleanly,
    _start_log_mirror,
    _stop_log_mirror,
    create_application,
    main,
)
from src.config import settings


@pytest.fixture(autouse=True)
def _remove_handlers_added_to_the_root_logger():
    # _start_log_mirror attaches its handler to the ROOT logger, which
    # persists for the life of the process - without this, a handler left
    # behind by one test keeps buffering every record logged by every test
    # that runs afterward.
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)


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

    assert app.post_shutdown is _shut_down_cleanly


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


@patch("src.bot.app._open_checkpointer", MagicMock())
@patch(
    "src.bot.app._load_retriever",
    MagicMock(side_effect=RuntimeError("chroma is gone")),
)
async def test_post_init_raises_startup_error_when_the_retriever_fails():
    # A bot that starts without a knowledge base would answer every question
    # with an apology; this is what stops it from starting at all.
    app = MagicMock()
    app.bot_data = {}

    with pytest.raises(StartupError):
        await _initialize_agents(app)


@patch("src.bot.app._open_checkpointer")
@patch(
    "src.bot.app._load_retriever",
    MagicMock(side_effect=RuntimeError("chroma is gone")),
)
async def test_post_init_leaves_the_checkpointer_in_bot_data_even_when_it_raises(
    mock_open_checkpointer,
):
    # Guards the moved assignment: post_shutdown reads bot_data["checkpointer"]
    # even after a failed startup, and can only close what it can find there.
    app = MagicMock()
    app.bot_data = {}

    with pytest.raises(StartupError):
        await _initialize_agents(app)

    assert app.bot_data["checkpointer"] is mock_open_checkpointer.return_value


def test_start_log_mirror_does_nothing_when_admin_chat_id_is_empty():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="")):
        app = MagicMock()
        app.bot_data = {}

        _start_log_mirror(app)

        assert "log_handler" not in app.bot_data


async def test_start_log_mirror_puts_the_handler_and_task_in_bot_data():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        app = MagicMock()
        app.bot_data = {}

        _start_log_mirror(app)

        try:
            assert {"log_handler", "log_mirror"} <= app.bot_data.keys()
        finally:
            app.bot_data["log_mirror"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.bot_data["log_mirror"]


async def test_start_log_mirror_attaches_the_handler_to_the_root_logger():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        app = MagicMock()
        app.bot_data = {}

        _start_log_mirror(app)

        try:
            assert app.bot_data["log_handler"] in logging.getLogger().handlers
        finally:
            app.bot_data["log_mirror"].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.bot_data["log_mirror"]


async def test_stop_log_mirror_cancels_and_awaits_the_task():
    # Only cancel() would leave the task merely scheduled to stop, which is
    # what left "Task was destroyed but it is pending" on the way out.
    async def _never_ends():
        await asyncio.sleep(100)

    app = MagicMock()
    app.bot_data = {"log_mirror": asyncio.create_task(_never_ends())}

    await _stop_log_mirror(app)

    assert app.bot_data["log_mirror"].done()


async def test_stop_log_mirror_detaches_the_handler_from_the_root_logger():
    # Left attached, a second _start_log_mirror later in the same process
    # would stack a second handler onto the first instead of replacing it.
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        app = MagicMock()
        app.bot_data = {}
        _start_log_mirror(app)

        await _stop_log_mirror(app)

        assert app.bot_data["log_handler"] not in logging.getLogger().handlers


async def test_stop_log_mirror_survives_a_startup_that_never_started_one():
    # _shut_down_cleanly runs even when post_init raised before
    # _start_log_mirror ran, so bot_data may hold no mirror task at all.
    app = MagicMock()
    app.bot_data = {}

    await _stop_log_mirror(app)


@patch("src.bot.app._close_checkpointer", new_callable=AsyncMock)
@patch("src.bot.app._stop_log_mirror", new_callable=AsyncMock)
async def test_shut_down_cleanly_stops_the_log_mirror_before_closing_the_checkpointer(
    mock_stop_mirror, mock_close_checkpointer
):
    # The mirror task can still be forwarding a record about the shutdown
    # itself; stopping it first means the checkpointer it might read from
    # is not pulled out from under it.
    manager = MagicMock()
    manager.attach_mock(mock_stop_mirror, "stop_mirror")
    manager.attach_mock(mock_close_checkpointer, "close_checkpointer")
    app = MagicMock()

    await _shut_down_cleanly(app)

    assert manager.mock_calls == [call.stop_mirror(app), call.close_checkpointer(app)]


def test_a_startup_error_escaping_run_polling_becomes_system_exit_1():
    # A plain sys.exit(1) call inside an except block would still exit the
    # test process with 0 once PTB's own signal handling gets involved; only
    # SystemExit reliably propagates as a failing exit code.
    mock_app = MagicMock()
    mock_app.run_polling = MagicMock(side_effect=StartupError("boom"))

    with patch("src.bot.app.create_application", return_value=mock_app):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
