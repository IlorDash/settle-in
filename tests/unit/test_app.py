import asyncio
import contextlib
import logging
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from telegram import Bot, BotCommandScopeChat, Update
from telegram.error import TelegramError
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

import src.bot.app  # noqa: F401  (imported for its logging configuration)
from src.bot.app import (
    HANDLED_UPDATES,
    OPERATOR_COMMANDS,
    PUBLIC_COMMANDS,
    StartupError,
    _close_checkpointer,
    _initialize_agents,
    _publish_operator_menu,
    _publish_public_menu,
    _run,
    _shut_down_cleanly,
    _start_log_mirror,
    _stop_log_mirror,
    create_application,
    main,
)
from src.bot.handlers import (
    ANNOUNCEMENT_PREFIX,
    announcement_callback,
    handle_edited_message,
)
from src.bot.middleware import KIND_PHOTO, KIND_TEXT
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


def _registered_command_names(app) -> set[str]:
    return {
        name
        for group in app.handlers.values()
        for handler in group
        if isinstance(handler, CommandHandler)
        for name in handler.commands
    }


def test_every_menu_command_is_registered_as_a_command_handler():
    # A name added to the menu tuples without a matching CommandHandler
    # would show the operator a command that does nothing when tapped.
    app = create_application()
    menu_names = {name for name, _ in PUBLIC_COMMANDS + OPERATOR_COMMANDS}

    assert menu_names <= _registered_command_names(app)


def test_every_non_public_command_handler_is_listed_as_an_operator_command():
    # A CommandHandler added to create_application() without a matching
    # OPERATOR_COMMANDS entry would work but never appear in anyone's menu.
    app = create_application()
    public_names = {name for name, _ in PUBLIC_COMMANDS}
    operator_names = {name for name, _ in OPERATOR_COMMANDS}

    assert _registered_command_names(app) - public_names <= operator_names


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
    #
    # admin_chat_id is pinned to empty so this does not depend on whatever
    # ADMIN_CHAT_ID happens to be set in the developer's own .env, and
    # set_my_commands is a real AsyncMock because the public menu is
    # published whether or not there is an operator: a plain MagicMock's
    # would fail with "object MagicMock can't be used in 'await'
    # expression" regardless of the orchestrator wiring under test.
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="")):
        app = MagicMock()
        app.bot_data = {}
        app.bot.set_my_commands = AsyncMock()

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


async def test_publish_public_menu_awaits_set_my_commands_once():
    bot = MagicMock(spec=Bot)
    bot.set_my_commands = AsyncMock()

    await _publish_public_menu(bot)

    bot.set_my_commands.assert_awaited_once()


async def test_publish_public_menu_names_no_scope():
    # No scope means Telegram's default one, which is what every chat
    # without a narrower scope of its own falls back to.
    bot = MagicMock(spec=Bot)
    bot.set_my_commands = AsyncMock()

    await _publish_public_menu(bot)

    assert "scope" not in bot.set_my_commands.call_args.kwargs


async def test_publish_public_menu_offers_no_operator_command():
    # The property the whole split exists for: an operator command must not
    # reach the menu every other user sees.
    bot = MagicMock(spec=Bot)
    bot.set_my_commands = AsyncMock()

    await _publish_public_menu(bot)

    published = bot.set_my_commands.call_args.args[0]
    names = [command.command for command in published]
    assert names == [name for name, _ in PUBLIC_COMMANDS]


async def test_publish_public_menu_swallows_a_telegram_error():
    # Nothing guards this call the way admin_chat_id guards the operator's,
    # so a Telegram outage at startup would raise straight out of post_init
    # and turn a cosmetic menu into exit code 1.
    bot = MagicMock(spec=Bot)
    bot.set_my_commands = AsyncMock(side_effect=TelegramError("nope"))

    await _publish_public_menu(bot)


async def test_publish_operator_menu_sets_no_menu_when_admin_chat_id_is_empty():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="")):
        bot = MagicMock(spec=Bot)
        bot.set_my_commands = AsyncMock()

        await _publish_operator_menu(bot)

        bot.set_my_commands.assert_not_called()


async def test_publish_operator_menu_awaits_set_my_commands_once():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        bot = MagicMock(spec=Bot)
        bot.set_my_commands = AsyncMock()

        await _publish_operator_menu(bot)

        bot.set_my_commands.assert_awaited_once()


async def test_publish_operator_menu_scopes_the_menu_to_the_admin_chat():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        bot = MagicMock(spec=Bot)
        bot.set_my_commands = AsyncMock()

        await _publish_operator_menu(bot)

        scope = bot.set_my_commands.call_args.kwargs["scope"]
        assert scope == BotCommandScopeChat("99")


async def test_publish_operator_menu_lists_public_commands_before_operator_ones():
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        bot = MagicMock(spec=Bot)
        bot.set_my_commands = AsyncMock()

        await _publish_operator_menu(bot)

        published = bot.set_my_commands.call_args.args[0]
        names = [command.command for command in published]
        assert names == [name for name, _ in PUBLIC_COMMANDS + OPERATOR_COMMANDS]


async def test_publish_operator_menu_swallows_a_telegram_error():
    # A menu is cosmetic; a chat Telegram refuses to set one for - the usual
    # cause is an ADMIN_CHAT_ID that has never messaged the bot - must not
    # raise out of startup and turn into exit code 1.
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        bot = MagicMock(spec=Bot)
        bot.set_my_commands = AsyncMock(side_effect=TelegramError("nope"))

        await _publish_operator_menu(bot)


@patch("src.bot.app.build_preference_tidier", MagicMock())
@patch("src.bot.app.build_translation_chain", MagicMock())
@patch("src.bot.app.build_rag_chain", MagicMock())
@patch("src.bot.app.build_orchestrator")
@patch("src.bot.app._open_checkpointer", MagicMock())
@patch("src.bot.app._load_retriever", MagicMock())
@patch("src.bot.app._start_log_mirror", MagicMock())
async def test_post_init_completes_when_the_operator_menu_cannot_be_set(mock_build):
    # The property the swallowed TelegramError exists for: a cosmetic menu
    # failure must not stop the rest of post_init from finishing.
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        app = MagicMock()
        app.bot_data = {}
        app.bot.set_my_commands = AsyncMock(side_effect=TelegramError("nope"))

        await _initialize_agents(app)

    assert app.bot_data["orchestrator"] is mock_build.return_value


@patch("src.bot.app.build_preference_tidier", MagicMock())
@patch("src.bot.app.build_translation_chain", MagicMock())
@patch("src.bot.app.build_rag_chain", MagicMock())
@patch("src.bot.app.build_orchestrator", MagicMock())
@patch("src.bot.app._open_checkpointer", MagicMock())
@patch("src.bot.app._load_retriever", MagicMock())
@patch("src.bot.app._start_log_mirror", MagicMock())
async def test_initialize_agents_publishes_both_the_public_and_the_operator_menu():
    # Each menu's own content and scope are already covered directly against
    # _publish_public_menu/_publish_operator_menu; this is the one thing
    # neither of those proves - that post_init actually reaches both calls.
    # Dropping the new call from _initialize_agents would leave every
    # non-admin chat without a "/" menu while every test of the function in
    # isolation stayed green.
    with patch("src.bot.app.settings", replace(settings, admin_chat_id="99")):
        app = MagicMock()
        app.bot_data = {}
        app.bot.set_my_commands = AsyncMock()

        await _initialize_agents(app)

    assert app.bot.set_my_commands.await_count == 2


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


def test_create_application_puts_a_daily_quota_in_bot_data():
    app = create_application()

    assert app.bot_data["daily_quota"] is not None


def test_create_application_daily_quota_carries_the_configured_text_limit():
    app = create_application()

    assert app.bot_data["daily_quota"].limits[KIND_TEXT] == settings.daily_text_limit


def test_create_application_daily_quota_carries_the_configured_photo_limit():
    app = create_application()

    assert app.bot_data["daily_quota"].limits[KIND_PHOTO] == settings.daily_photo_limit


def test_create_application_registers_a_limits_command_handler():
    app = create_application()

    assert "limits" in _registered_command_names(app)


def test_limits_is_offered_to_everyone_not_only_the_operator():
    assert "limits" in {name for name, _ in PUBLIC_COMMANDS}


def test_limits_is_not_repeated_in_the_operator_commands():
    # The operator's menu is the two tuples concatenated, so a command in
    # both would appear twice in their "/" list.
    assert "limits" not in {name for name, _ in OPERATOR_COMMANDS}


def test_create_application_registers_the_edited_message_handler_first():
    # Order is load-bearing: python-telegram-bot runs only the first handler
    # in a group that matches an update, and every handler below this one
    # reads update.message, which an edit update leaves as None.
    app = create_application()

    first = app.handlers[0][0]

    assert isinstance(first, MessageHandler) and first.callback is handle_edited_message


def test_handled_updates_includes_ordinary_messages():
    # Commands and the text/photo/file handlers all key off Update.MESSAGE.
    assert Update.MESSAGE in HANDLED_UPDATES


def test_handled_updates_includes_edited_messages():
    # handle_edited_message is registered to answer Update.EDITED_MESSAGE;
    # without it in the allow list Telegram would never deliver the edit.
    assert Update.EDITED_MESSAGE in HANDLED_UPDATES


def test_handled_updates_includes_callback_queries():
    # The feedback, large-file, operator-panel and announcement keyboards
    # all arrive as Update.CALLBACK_QUERY.
    assert Update.CALLBACK_QUERY in HANDLED_UPDATES


def test_handled_updates_excludes_channel_posts():
    # The bot is an administrator of the announcements channel, so Telegram
    # sends every post made there as Update.CHANNEL_POST. Left in the allow
    # list, the operator's own post in their channel would reach
    # handle_message as ordinary text.
    assert Update.CHANNEL_POST not in HANDLED_UPDATES


def test_handled_updates_excludes_edited_channel_posts():
    # Same hole as a fresh channel post, but for an edit of one.
    assert Update.EDITED_CHANNEL_POST not in HANDLED_UPDATES


def test_run_polls_with_the_handled_updates_allow_list():
    with patch("src.bot.app.settings", replace(settings, bot_mode="polling")):
        app = MagicMock()

        _run(app)

        app.run_polling.assert_called_once_with(allowed_updates=HANDLED_UPDATES)


def test_run_serves_a_webhook_with_the_handled_updates_allow_list():
    with patch(
        "src.bot.app.settings",
        replace(settings, bot_mode="webhook", webhook_url="https://example.com/hook"),
    ):
        app = MagicMock()

        _run(app)

        assert app.run_webhook.call_args.kwargs["allowed_updates"] == HANDLED_UPDATES


def test_create_application_registers_the_announcement_callback_handler():
    app = create_application()
    handlers = [
        handler
        for group in app.handlers.values()
        for handler in group
        if isinstance(handler, CallbackQueryHandler)
        and handler.callback is announcement_callback
    ]

    assert handlers[0].pattern.pattern == f"^{ANNOUNCEMENT_PREFIX}:"


@patch("src.bot.app.build_preference_tidier", MagicMock())
@patch("src.bot.app.build_translation_chain", MagicMock())
@patch("src.bot.app.build_rag_chain", MagicMock())
@patch("src.bot.app.build_orchestrator")
@patch("src.bot.app._open_checkpointer", MagicMock())
@patch("src.bot.app._load_retriever", MagicMock())
@patch("src.bot.app._start_log_mirror", MagicMock())
async def test_post_init_completes_when_the_channel_probe_fails(mock_build):
    # check_channel_access already swallows TelegramError itself; this
    # guards the wiring one level up - a probe that could not reach the
    # channel must not stop the rest of post_init from finishing.
    with (
        patch(
            "src.bot.app.settings",
            replace(settings, admin_chat_id="", announcement_channel="settlein_news"),
        ),
        patch(
            "src.bot.channel.settings",
            replace(settings, admin_chat_id="", announcement_channel="settlein_news"),
        ),
    ):
        app = MagicMock()
        app.bot_data = {}
        app.bot.set_my_commands = AsyncMock()
        app.bot.get_chat_member = AsyncMock(side_effect=TelegramError("nope"))

        await _initialize_agents(app)

    assert app.bot_data["orchestrator"] is mock_build.return_value


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
