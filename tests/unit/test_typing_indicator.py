import asyncio
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ChatAction
from telegram.error import BadRequest, Forbidden

from src.bot.typing_indicator import show_typing


async def test_the_indicator_is_sent_on_entering_the_block(mock_bot):
    async with show_typing(mock_bot, 123):
        pass

    mock_bot.send_chat_action.assert_awaited_once_with(
        chat_id=123, action=ChatAction.TYPING
    )


async def test_the_refresher_stops_once_the_block_exits(mock_bot, monkeypatch):
    # A tiny refresh interval lets several refreshes happen inside the block
    # without the test itself waiting seconds.
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)

    async with show_typing(mock_bot, 123):
        await asyncio.sleep(0.05)
    count_at_exit = mock_bot.send_chat_action.await_count
    await asyncio.sleep(0.05)

    assert mock_bot.send_chat_action.await_count == count_at_exit


async def test_the_refresher_actually_ran_before_the_block_exited(
    mock_bot, monkeypatch
):
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)

    async with show_typing(mock_bot, 123):
        await asyncio.sleep(0.05)

    # One send for entering the block, plus at least one refresh.
    assert mock_bot.send_chat_action.await_count > 1


async def test_the_indicator_stops_when_the_agent_raises(mock_bot):
    with pytest.raises(RuntimeError):
        async with show_typing(mock_bot, 123):
            raise RuntimeError("agent failed")


async def test_the_refresher_stops_after_the_block_raises(mock_bot, monkeypatch):
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)

    with pytest.raises(RuntimeError):
        async with show_typing(mock_bot, 123):
            await asyncio.sleep(0.05)
            raise RuntimeError("agent failed")
    count_at_exit = mock_bot.send_chat_action.await_count
    await asyncio.sleep(0.05)

    assert mock_bot.send_chat_action.await_count == count_at_exit


async def test_a_forbidden_chat_starts_no_refresher(mock_bot, monkeypatch):
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)
    mock_bot.send_chat_action = AsyncMock(side_effect=Forbidden("bot was blocked"))

    async with show_typing(mock_bot, 123):
        await asyncio.sleep(0.05)

    assert mock_bot.send_chat_action.await_count == 1


async def test_a_bad_request_chat_starts_no_refresher(mock_bot, monkeypatch):
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)
    mock_bot.send_chat_action = AsyncMock(side_effect=BadRequest("chat not found"))

    async with show_typing(mock_bot, 123):
        await asyncio.sleep(0.05)

    assert mock_bot.send_chat_action.await_count == 1


async def test_the_block_still_runs_when_the_chat_is_forbidden(mock_bot):
    mock_bot.send_chat_action = AsyncMock(side_effect=Forbidden("bot was blocked"))
    ran = False

    async with show_typing(mock_bot, 123):
        ran = True

    assert ran


async def test_the_refresher_stops_when_a_later_send_is_forbidden(
    mock_bot, monkeypatch
):
    # The first send (before the block starts) succeeds, so the refresher is
    # started; a refusal on its own send must stop the loop just the same.
    monkeypatch.setattr("src.bot.typing_indicator.TYPING_REFRESH_SECONDS", 0.01)
    mock_bot.send_chat_action = AsyncMock(side_effect=[None, Forbidden("blocked")])

    async with show_typing(mock_bot, 123):
        await asyncio.sleep(0.05)
    count_after_the_refusal = mock_bot.send_chat_action.await_count
    await asyncio.sleep(0.05)

    assert mock_bot.send_chat_action.await_count == count_after_the_refusal


async def test_a_non_telegram_error_does_not_escape_the_context_manager(mock_bot):
    # python-telegram-bot reports a bot that is not yet initialised as a
    # plain RuntimeError, not a TelegramError - a blip worth retrying, and
    # it must never cost the caller the answer the model already paid for.
    mock_bot.send_chat_action = AsyncMock(
        side_effect=RuntimeError("Bot is not properly initialized")
    )

    async with show_typing(mock_bot, 123):
        pass
