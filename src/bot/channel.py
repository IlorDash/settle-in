"""Post the bot's announcements to a public channel users subscribe to.

A channel rather than a mailing to every chat, and the difference is not
cosmetic. Messaging each chat in turn is lossy - Telegram will not deliver to
someone who blocked the bot or never started it, and nothing is queued for a
user who is offline - and it is rate-limited to roughly thirty messages a
second across all chats, so a real user base means minutes of paced sending
while ordinary replies compete with the mailing. One message to a channel has
neither problem: Telegram does the fan-out, and anyone can read it by opening
the link whether or not they have ever talked to the bot. The cost is that it
is opt-in, which is why /start and /help carry the link.

The operator is the channel's administrator and writes in it by hand as well.
The bot is therefore one poster among others: it only ever sends, and never
reads, edits or deletes anything already there.
"""

import logging

from telegram import Bot
from telegram.constants import ChatMemberStatus
from telegram.error import BadRequest, Forbidden, TelegramError

from src.config import settings

logger = logging.getLogger(__name__)

CHANNEL_LINK = "https://t.me/{name}"

# What the operator reads when a post did not happen. Each names the channel,
# because the likeliest cause of all three is that ANNOUNCEMENT_CHANNEL points
# somewhere other than where they think it does.
NOT_CONFIGURED = (
    "There is no announcement channel set up, so there is nowhere to post. "
    "Set ANNOUNCEMENT_CHANNEL and restart the bot."
)
NOT_ADMINISTRATOR = (
    "I am not an administrator of {name}, so Telegram will not let me post "
    "there. Add me to the channel as an administrator."
)
CHANNEL_UNKNOWN = "Telegram does not know a channel called {name}."
POST_FAILED = "Telegram would not accept the announcement for {name}."


def channel_name() -> str | None:
    """Name the announcement channel the way Telegram wants it addressed.

    The `@` is added when it is missing rather than demanded of the operator:
    without it every send fails with a message nobody would connect back to a
    character left out of an environment variable.

    Returns:
        The channel as "@name", or None when none is configured.
    """
    configured = settings.announcement_channel.strip()
    if not configured:
        return None
    return f"@{configured.lstrip('@')}"


async def post_announcement(bot: Bot, text: str) -> str | None:
    """Publish one announcement, and say in words when it could not be.

    Never raises. The callers are handlers and button taps, where an escaping
    TelegramError would become an apology about the AI service - which is not
    what went wrong - or an unhandled error with no explanation at all.

    Failures are logged at WARNING rather than ERROR on purpose: the caller
    replies to the operator with the same reason in their own chat, and ERROR
    would have TelegramLogHandler mirror a second copy of a problem already on
    their screen. The log line is the record; the reply is the report.

    Args:
        bot: The bot to post with.
        text: The announcement, as subscribers will read it.

    Returns:
        None when the announcement was posted, or a line explaining to the
        operator why it was not.
    """
    name = channel_name()
    if name is None:
        return NOT_CONFIGURED

    try:
        await bot.send_message(chat_id=name, text=text)
    except Forbidden:
        logger.warning("Not an administrator of announcement channel %s.", name)
        return NOT_ADMINISTRATOR.format(name=name)
    except BadRequest:
        logger.warning("Announcement channel %s does not exist.", name)
        return CHANNEL_UNKNOWN.format(name=name)
    except TelegramError as error:
        logger.warning("Could not post an announcement to %s: %s", name, error)
        return POST_FAILED.format(name=name)

    logger.info("Posted an announcement to %s.", name)
    return None


async def check_channel_access(bot: Bot) -> None:
    """Warn at startup if announcements would fail, without stopping the bot.

    Called from the post_init hook, so it logs and lets go for the same reason
    the operator menu does: an exception there becomes exit code 1, and a
    channel is not worth refusing to start over.

    It asks for the bot's own membership rather than for the channel, because
    get_chat succeeds on any public channel and would miss the likeliest
    mistake of the two - the bot added to the channel but never made an
    administrator, which reads as fine until the first announcement.

    Args:
        bot: The bot whose access is being checked.
    """
    name = channel_name()
    if name is None:
        return

    try:
        membership = await bot.get_chat_member(chat_id=name, user_id=bot.id)
    except TelegramError as error:
        logger.warning("Cannot reach announcement channel %s: %s", name, error)
        return

    if membership.status != ChatMemberStatus.ADMINISTRATOR:
        logger.warning(
            "Not an administrator of announcement channel %s (status: %s); "
            "announcements will fail.",
            name,
            membership.status,
        )
        return

    logger.info("Announcements will be posted to %s.", name)
