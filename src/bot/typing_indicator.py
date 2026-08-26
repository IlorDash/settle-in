"""Telegram's "typing…" indicator, kept alive while an agent is working.

Telegram clears a chat action a few seconds after it is sent, so a single
call is not enough to cover a reply that takes an LLM call or two. The
context manager here re-sends the action on a loop and stops the moment the
block it wraps finishes, whether that is an answer or an exception.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from telegram import Bot
from telegram.constants import ChatAction
from telegram.error import BadRequest, Forbidden

logger = logging.getLogger(__name__)

# Telegram drops the indicator about five seconds after send_chat_action.
# Refreshing a second early keeps the chat header from flickering between
# the expiry and the next send.
TYPING_REFRESH_SECONDS = 4.0


@asynccontextmanager
async def show_typing(bot: Bot, chat_id: int) -> AsyncIterator[None]:
    """Show "typing…" in the chat header for as long as the block runs.

    The first action is sent before the block starts, so the indicator is
    already up when the slow work begins. A background task then re-sends it
    until the block exits — on an answer or on an exception, since the stop
    lives in a `finally`.

    Args:
        bot: The bot to send the chat action with.
        chat_id: The chat to show the indicator in.

    Yields:
        None. The caller does the slow work inside the block.
    """
    if not await _send_typing(bot, chat_id):
        # The chat refuses the action outright, so a refresher would do
        # nothing but spend a request every few seconds.
        yield
        return

    refresher = asyncio.create_task(_refresh_typing(bot, chat_id))
    try:
        yield
    finally:
        refresher.cancel()
        # Awaited so no refresh lands after the answer and leaves the header
        # claiming the bot is still typing. gather(return_exceptions=True)
        # rather than a bare await: it does not raise the cancellation we
        # just asked for, and it does not swallow one aimed at the handler.
        await asyncio.gather(refresher, return_exceptions=True)


async def _send_typing(bot: Bot, chat_id: int) -> bool:
    """Send the typing action once, saying whether it is worth sending again.

    Args:
        bot: The bot to send the chat action with.
        chat_id: The chat to show the indicator in.

    Returns:
        False when this chat will never accept the action, True when it was
        sent or failed in a way that may well work on the next try.
    """
    try:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except (Forbidden, BadRequest):
        # The bot is blocked or the chat is gone: retrying costs a request
        # every few seconds and will never succeed.
        logger.debug("Chat %s will not accept a typing indicator", chat_id)
        return False
    except Exception:
        # The indicator is decoration, so nothing here may reach the caller.
        # A dropped connection - or a bot that is not initialised, which
        # python-telegram-bot reports as a plain RuntimeError - must not cost
        # the user an answer the model has already been paid to write. A
        # blip is worth retrying, so this is not a refusal.
        logger.debug("Could not send the typing indicator to chat %s", chat_id)
    return True


async def _refresh_typing(bot: Bot, chat_id: int) -> None:
    """Re-send the typing action until cancelled, or until the chat refuses it.

    Args:
        bot: The bot to send the chat action with.
        chat_id: The chat to show the indicator in.
    """
    while True:
        await asyncio.sleep(TYPING_REFRESH_SECONDS)
        if not await _send_typing(bot, chat_id):
            return
