import asyncio
import logging
import sys
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.admin import (
    BUFFER_SIZE,
    MAX_RECORDS_PER_MESSAGE,
    TelegramLogHandler,
    is_admin,
    mirror_logs,
)
from src.config import settings


def _record(
    name="src.bot.app",
    level=logging.ERROR,
    msg="something happened",
    args=(),
    exc_info=None,
):
    """Build a LogRecord without going through the logging module's registry."""
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


def test_an_httpx_record_is_not_buffered():
    handler = TelegramLogHandler()

    handler.handle(_record(name="httpx"))

    assert handler.snapshot() == []


def test_an_httpx_record_is_not_queued():
    handler = TelegramLogHandler()

    handler.handle(_record(name="httpx"))

    assert handler.take_pending() == []


def test_a_telegram_request_record_is_not_buffered():
    handler = TelegramLogHandler()

    handler.handle(_record(name="telegram.request.HTTPXRequest"))

    assert handler.snapshot() == []


def test_a_telegram_ext_record_is_buffered():
    # The guard for the narrowed filter: `telegram.ext` is where PTB logs
    # "Conflict: terminated by other getUpdates request", which the operator
    # needs to see. Broadening IGNORED_LOGGERS to "telegram" would drop it.
    handler = TelegramLogHandler()

    handler.handle(_record(name="telegram.ext.Application"))

    assert handler.snapshot() != []


def test_a_telegram_ext_record_is_queued():
    handler = TelegramLogHandler()

    handler.handle(_record(name="telegram.ext.Application"))

    assert handler.take_pending() != []


def test_two_records_sharing_a_template_queue_only_once():
    handler = TelegramLogHandler()

    handler.handle(_record(msg="turning away user %s", args=("12",)))
    handler.handle(_record(msg="turning away user %s", args=("34",)))

    assert len(handler.take_pending()) == 1


def test_a_different_template_queues_separately():
    handler = TelegramLogHandler()

    handler.handle(_record(msg="turning away user %s", args=("12",)))
    handler.handle(_record(msg="a completely different problem"))

    assert len(handler.take_pending()) == 2


def test_an_info_record_is_buffered_by_default():
    handler = TelegramLogHandler()

    handler.handle(_record(level=logging.INFO))

    assert handler.snapshot() != []


def test_an_info_record_is_not_queued_at_the_default_push_level():
    handler = TelegramLogHandler()

    handler.handle(_record(level=logging.INFO))

    assert handler.take_pending() == []


def test_lowering_the_push_level_to_warning_queues_a_warning():
    handler = TelegramLogHandler()
    handler.push_level = logging.WARNING

    handler.handle(_record(level=logging.WARNING))

    assert handler.take_pending() != []


def test_take_pending_drains_the_queue():
    handler = TelegramLogHandler()
    handler.handle(_record())

    handler.take_pending()

    assert handler.take_pending() == []


def test_take_pending_returns_at_most_max_records_per_message():
    handler = TelegramLogHandler()
    for i in range(MAX_RECORDS_PER_MESSAGE + 3):
        handler.handle(_record(msg=f"distinct problem {i}"))

    assert len(handler.take_pending()) == MAX_RECORDS_PER_MESSAGE


def test_recent_is_capped_at_buffer_size():
    handler = TelegramLogHandler()
    for i in range(BUFFER_SIZE + 10):
        handler.handle(_record(level=logging.INFO, msg=f"line {i}"))

    assert len(handler.snapshot()) == BUFFER_SIZE


def test_a_record_whose_formatting_raises_does_not_propagate_to_the_caller():
    # This handler sits on the root logger, and handle() does not guard
    # emit() - without the try/except in emit(), a single mis-called
    # logger.error(...) anywhere in the app would itself raise.
    logger = logging.getLogger("test-admin-format-error")
    logger.propagate = False
    handler = TelegramLogHandler()
    logger.addHandler(handler)
    try:
        logger.error("two args: %s %s", "only-one")
    finally:
        logger.removeHandler(handler)


def test_the_pushed_line_for_a_record_with_exc_info_contains_the_exception():
    # Built from getMessage(), not a slice of the formatted line - a
    # formatter puts the traceback last, so a head-slice would cut it away.
    handler = TelegramLogHandler()
    try:
        raise ValueError("bad token")
    except ValueError:
        record = _record(msg="something broke", exc_info=sys.exc_info())

    handler.handle(record)

    assert "ValueError: bad token" in handler.take_pending()[0]


async def test_mirror_logs_sends_the_batch_to_the_admin_chat():
    handler = TelegramLogHandler()
    handler.handle(_record(name="test.mod", msg="boom"))
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with (
        patch("src.bot.admin.settings", replace(settings, admin_chat_id="99")),
        patch(
            "src.bot.admin.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await mirror_logs(bot, handler)

    bot.send_message.assert_awaited_once_with(chat_id="99", text="ERROR test.mod: boom")


async def test_a_send_failure_does_not_end_the_mirror_loop():
    handler = TelegramLogHandler()
    handler.handle(_record())
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=RuntimeError("telegram is down"))

    with (
        patch("src.bot.admin.settings", replace(settings, admin_chat_id="99")),
        patch(
            "src.bot.admin.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        ),
    ):
        # The second sleep() is only reached if the RuntimeError from the
        # first send did not escape mirror_logs and end the loop.
        with pytest.raises(asyncio.CancelledError):
            await mirror_logs(bot, handler)


async def test_mirror_logs_sends_nothing_when_there_is_nothing_pending():
    handler = TelegramLogHandler()
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with (
        patch("src.bot.admin.settings", replace(settings, admin_chat_id="99")),
        patch(
            "src.bot.admin.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await mirror_logs(bot, handler)

    bot.send_message.assert_not_awaited()


def test_is_admin_true_for_the_configured_chat():
    with patch("src.bot.admin.settings", replace(settings, admin_chat_id="99")):
        assert is_admin(99) is True


def test_is_admin_false_for_a_different_chat():
    with patch("src.bot.admin.settings", replace(settings, admin_chat_id="99")):
        assert is_admin(100) is False


def test_is_admin_false_when_admin_chat_id_is_empty():
    with patch("src.bot.admin.settings", replace(settings, admin_chat_id="")):
        assert is_admin(99) is False
