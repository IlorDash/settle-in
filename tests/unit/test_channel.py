import logging
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Bot
from telegram.constants import ChatMemberStatus
from telegram.error import BadRequest, Forbidden, TelegramError

from src.bot.channel import (
    CHANNEL_UNKNOWN,
    NOT_ADMINISTRATOR,
    NOT_CONFIGURED,
    POST_FAILED,
    channel_link,
    channel_name,
    check_channel_access,
    post_announcement,
)
from src.config import settings


def test_channel_name_adds_the_at_sign_when_missing():
    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        assert channel_name() == "@settlein_news"


def test_channel_name_leaves_the_at_sign_alone_when_already_present():
    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="@settlein_news"),
    ):
        assert channel_name() == "@settlein_news"


def test_channel_name_is_none_when_the_setting_is_empty():
    with patch("src.bot.channel.settings", replace(settings, announcement_channel="")):
        assert channel_name() is None


def test_channel_name_is_none_when_the_setting_is_whitespace():
    with patch(
        "src.bot.channel.settings", replace(settings, announcement_channel="   ")
    ):
        assert channel_name() is None


def test_channel_link_gives_the_t_me_url():
    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        assert channel_link() == "https://t.me/settlein_news"


def test_channel_link_is_none_when_unset():
    with patch("src.bot.channel.settings", replace(settings, announcement_channel="")):
        assert channel_link() is None


async def test_post_announcement_returns_none_on_success():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        result = await post_announcement(bot, "hello")

    assert result is None


async def test_post_announcement_sends_to_the_normalised_channel_name():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        await post_announcement(bot, "hello")

    bot.send_message.assert_awaited_once_with(chat_id="@settlein_news", text="hello")


async def test_post_announcement_returns_a_reason_when_not_an_administrator():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock(side_effect=Forbidden("not admin"))

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        result = await post_announcement(bot, "hello")

    assert result == NOT_ADMINISTRATOR.format(name="@settlein_news")


async def test_post_announcement_returns_a_reason_when_the_channel_does_not_exist():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock(side_effect=BadRequest("no such channel"))

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        result = await post_announcement(bot, "hello")

    assert result == CHANNEL_UNKNOWN.format(name="@settlein_news")


async def test_post_announcement_returns_a_reason_for_any_other_telegram_error():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock(side_effect=TelegramError("nope"))

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        result = await post_announcement(bot, "hello")

    assert result == POST_FAILED.format(name="@settlein_news")


async def test_post_announcement_returns_not_configured_when_the_channel_is_unset():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    with patch("src.bot.channel.settings", replace(settings, announcement_channel="")):
        result = await post_announcement(bot, "hello")

    assert result == NOT_CONFIGURED


async def test_post_announcement_never_calls_send_message_when_the_channel_is_unset():
    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    with patch("src.bot.channel.settings", replace(settings, announcement_channel="")):
        await post_announcement(bot, "hello")

    bot.send_message.assert_not_awaited()


async def test_check_channel_access_calls_telegram_nothing_when_unset():
    bot = MagicMock(spec=Bot)
    bot.get_chat_member = AsyncMock()

    with patch("src.bot.channel.settings", replace(settings, announcement_channel="")):
        await check_channel_access(bot)

    bot.get_chat_member.assert_not_awaited()


async def test_check_channel_access_does_not_raise_when_telegram_errors():
    bot = MagicMock(spec=Bot)
    bot.get_chat_member = AsyncMock(side_effect=TelegramError("nope"))

    with patch(
        "src.bot.channel.settings",
        replace(settings, announcement_channel="settlein_news"),
    ):
        await check_channel_access(bot)


async def test_check_channel_access_warns_when_not_an_administrator(caplog):
    bot = MagicMock(spec=Bot)
    membership = MagicMock()
    membership.status = ChatMemberStatus.MEMBER
    bot.get_chat_member = AsyncMock(return_value=membership)

    with (
        patch(
            "src.bot.channel.settings",
            replace(settings, announcement_channel="settlein_news"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        await check_channel_access(bot)

    assert "not an administrator" in caplog.text.lower()


async def test_check_channel_access_is_quiet_for_a_proper_administrator(caplog):
    bot = MagicMock(spec=Bot)
    membership = MagicMock()
    membership.status = ChatMemberStatus.ADMINISTRATOR
    bot.get_chat_member = AsyncMock(return_value=membership)

    with (
        patch(
            "src.bot.channel.settings",
            replace(settings, announcement_channel="settlein_news"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        await check_channel_access(bot)

    assert caplog.text == ""
