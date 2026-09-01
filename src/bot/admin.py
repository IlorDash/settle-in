"""Mirror the bot's problems into the operator's chat, without spamming them.

Some failures no user can do anything about: the OpenAI credit runs out, or
the API key stops being accepted. Every message from then on fails the same
way, so forwarding each log record would send one Telegram message per
message received. Each distinct record is therefore forwarded at most once
per run of the process, keyed by the log template rather than the finished
line - "turning away user 12" and "turning away user 34" are the same
problem reported twice.

The chat is a notification channel, not a log viewer: only the level set on
the handler and above is pushed, while everything the bot logs is kept in a
small in-memory buffer that `/logs` sends back as a file. After a restart
that buffer is empty, which is the point at which the server's own logs are
the place to look.
"""

import asyncio
import logging
import traceback
from collections import deque

from telegram import Bot

from src.config import settings

logger = logging.getLogger(__name__)

# Records kept for /logs. Two hundred lines is a few minutes of a busy chat
# and costs nothing; the volume-backed log on the server is the long record.
BUFFER_SIZE = 200

# Delivering a record makes network calls, and those calls log. Forwarding
# their logs would call them again, so the loggers on the delivery path are
# dropped before anything else happens. This is not noise reduction: without
# it the first forwarded error never stops.
#
# Only the delivery path. `telegram.ext` is deliberately absent: that is
# where "Conflict: terminated by other getUpdates request" is logged, which
# is exactly what an operator needs to be told about.
IGNORED_LOGGERS = ("httpx", "httpcore", "telegram.request", "telegram.Bot", __name__)

# How long records wait before being sent. A batch keeps a burst of errors -
# and errors do arrive in bursts, since one outage fails every request at
# once - inside Telegram's roughly one-message-per-second limit for a chat.
FLUSH_SECONDS = 5.0

# Five records of six hundred characters stay under Telegram's message cap
# without this module having to restate what that cap is.
MAX_RECORDS_PER_MESSAGE = 5
MAX_RECORD_CHARS = 600

# A library that pre-formats its messages gives every record a distinct
# template, so the set of seen templates needs a ceiling. Reached only by
# such a logger; the bot's own templates number a few dozen.
MAX_REPORTED_TEMPLATES = 500

LOG_LEVELS = {"warning": logging.WARNING, "error": logging.ERROR}

# Shorter than the server's format: a phone screen is narrow, and the date
# is already on the Telegram message.
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _is_on_delivery_path(name: str) -> bool:
    """Say whether a logger is one that sending a message would trigger."""
    return any(
        name == ignored or name.startswith(f"{ignored}.") for ignored in IGNORED_LOGGERS
    )


class TelegramLogHandler(logging.Handler):
    """Keep recent log records, and queue the serious ones for the operator.

    `emit` is called synchronously and from whichever thread logged, so it
    only ever appends. Sending is left to `mirror_logs`, which runs on the
    event loop.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(LOG_FORMAT))
        self.push_level = logging.ERROR
        self._recent: deque[str] = deque(maxlen=BUFFER_SIZE)
        self._pending: list[str] = []
        # Insertion-ordered, so the oldest template is the one evicted.
        self._reported: dict[tuple[str, str], None] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Drop the records that delivering a record would itself produce."""
        return not _is_on_delivery_path(record.name)

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer every record, and queue the ones the operator should see.

        This handler sits on the root logger, so a failure here would
        surface inside whatever called `logger.error(...)` - `handle()` does
        not guard `emit()`. Anything that goes wrong is reported the way the
        standard handlers report it and nothing is raised at the caller.
        """
        try:
            self._recent.append(self.format(record))
            if record.levelno < self.push_level or self._seen_before(record):
                return
            self._pending.append(self._notification(record))
        except Exception:
            self.handleError(record)

    def take_pending(self) -> list[str]:
        """Hand over the queued records and forget them."""
        with self.lock:
            batch = self._pending[:MAX_RECORDS_PER_MESSAGE]
            del self._pending[: len(batch)]
        return batch

    def snapshot(self) -> list[str]:
        """Copy the buffered records, for /logs to send as a file."""
        with self.lock:
            return list(self._recent)

    def _seen_before(self, record: logging.LogRecord) -> bool:
        """Say whether this log template has already been reported."""
        key = (record.name, str(record.msg))
        if key in self._reported:
            return True
        self._reported[key] = None
        if len(self._reported) > MAX_REPORTED_TEMPLATES:
            self._reported.pop(next(iter(self._reported)))
        return False

    def _notification(self, record: logging.LogRecord) -> str:
        """Render the short line the operator reads on their phone.

        Built from the message rather than sliced off the formatted record:
        a formatter puts the traceback last, so truncating the formatted
        string would cut away the exception itself - the one part worth
        telling somebody about.
        """
        line = f"{record.levelname} {record.name}: {record.getMessage()}"
        if record.exc_info:
            reason = traceback.format_exception_only(record.exc_info[1])[-1]
            line = f"{line}\n{reason.strip()}"
        return line[:MAX_RECORD_CHARS]


def is_admin(chat_id: int) -> bool:
    """Say whether a chat is the operator's, so a command may run there."""
    return bool(settings.admin_chat_id) and str(chat_id) == settings.admin_chat_id


async def mirror_logs(bot: Bot, handler: TelegramLogHandler) -> None:
    """Forward queued records to the operator until cancelled.

    Runs as a background task for the life of the bot. Every failure is
    swallowed, not just a Telegram one: this task exists to report problems,
    so it must neither become one nor stop reporting because a single send
    went wrong. `CancelledError` is a BaseException, so shutdown still ends
    the loop.

    Args:
        bot: The bot to send with.
        handler: The handler collecting the records.
    """
    while True:
        await asyncio.sleep(FLUSH_SECONDS)
        batch = handler.take_pending()
        if not batch:
            continue
        try:
            await bot.send_message(
                chat_id=settings.admin_chat_id, text="\n\n".join(batch)
            )
        except Exception:
            # Logged under this module's name, which the handler's own
            # filter drops - otherwise a failing send would queue a record
            # about failing to send.
            logger.warning(
                "Could not deliver operator logs to chat %s.", settings.admin_chat_id
            )
