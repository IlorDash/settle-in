"""Middleware utilities for input validation and rate limiting.

These functions act as guards that run before the main handler logic.
They protect against invalid input and excessive usage.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone

MAX_MESSAGE_LENGTH = 1000
RATE_LIMIT_MESSAGES = 5
RATE_LIMIT_WINDOW_SECONDS = 60

# The two things a chat can spend in a day. Separate buckets rather than one
# budget a photo draws more from: a user whose photos are gone can then be
# told plainly that typed questions still work.
KIND_TEXT = "text"
KIND_PHOTO = "photo"
QUOTA_KINDS = (KIND_TEXT, KIND_PHOTO)

QUOTA_EXHAUSTED = {
    KIND_TEXT: (
        "You have used your {limit} questions for today. The allowance "
        "resets at midnight UTC - I'll be here then."
    ),
    KIND_PHOTO: (
        "You have used your {limit} photos for today. Reading a photo costs "
        "several times what a typed question does, so it has its own smaller "
        "allowance - questions still work, and the photos come back at "
        "midnight UTC."
    ),
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024
# Above this an upload is worth asking the sender about before reading. The
# vision model is billed in 32x32 patches of the image it is given, and
# detail="original" hands it the file untouched: a full-size phone photo sent
# as a document costs several times the compressed copy Telegram makes when
# the same picture is sent as a photo. Bytes are a rough stand-in for pixels,
# but they are the only size Telegram reports before the file is downloaded.
LARGE_IMAGE_BYTES = 2 * 1024 * 1024
# The formats the OpenAI vision models accept. An iPhone HEIC sent as a file
# lands here, and a clear refusal beats the API's own error.
SUPPORTED_IMAGE_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


class ValidationError(Exception):
    """Raised when user input fails validation.

    Attributes:
        user_message: Friendly message to show the user.
    """

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


def validate_message_text(text: str | None) -> str:
    """Validate and sanitize user message text.

    Checks that the text is not empty and within the allowed length.
    Strips leading/trailing whitespace.

    Args:
        text: Raw message text from the user (may be None).

    Returns:
        Cleaned message text, stripped of extra whitespace.

    Raises:
        ValidationError: If the text is empty or exceeds MAX_MESSAGE_LENGTH.
    """
    if not text or not text.strip():
        raise ValidationError(
            "It looks like you sent an empty message. "
            "Please type a question and I'll try to help!"
        )

    cleaned = text.strip()

    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValidationError(
            f"Your message is too long ({len(cleaned)} characters). "
            f"Please keep it under {MAX_MESSAGE_LENGTH} characters."
        )

    return cleaned


def validate_image_upload(size_bytes: int | None, mime_type: str) -> None:
    """Check that an uploaded image is one we can afford to send to the model.

    Telegram reports the size and type in the update itself, so this runs
    before the download and rejects an oversized file without fetching it.
    This is the hard limit; LARGE_IMAGE_BYTES is the softer one above which
    the sender is asked first.

    Args:
        size_bytes: The file's size as Telegram reports it, or None when it
            does not, in which case the size is left to Telegram's own limit.
        mime_type: The file's media type, e.g. "image/jpeg".

    Raises:
        ValidationError: If the file is too large or not a supported image.
    """
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValidationError(
            "I can only read JPEG, PNG, GIF, and WebP images. "
            "Try sending the document as a normal photo."
        )

    if size_bytes is not None and size_bytes > MAX_IMAGE_BYTES:
        megabytes = MAX_IMAGE_BYTES // (1024 * 1024)
        raise ValidationError(
            f"That image is too large. Please send one under {megabytes} MB."
        )


def is_large_upload(size_bytes: int | None) -> bool:
    """Tell whether an upload is big enough to be worth a word to the sender.

    Not a rejection, a question: the handler offers the sender the choice
    rather than reading straight away. A full-size photo sent as a file is
    read at its own resolution and costs several times the copy Telegram
    makes when the same picture is sent as a photo, and the sender has no
    way of knowing that.

    Args:
        size_bytes: The file's size as Telegram reports it, or None when it
            does not report one.

    Returns:
        True if the upload is over LARGE_IMAGE_BYTES.
    """
    return size_bytes is not None and size_bytes > LARGE_IMAGE_BYTES


class RateLimiter:
    """Tracks message timestamps per user to enforce rate limits.

    Uses a sliding window approach: only messages within the last
    RATE_LIMIT_WINDOW_SECONDS count toward the limit.

    Attributes:
        max_messages: Maximum messages allowed per window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(
        self,
        max_messages: int = RATE_LIMIT_MESSAGES,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self._timestamps: dict[int, list[float]] = {}

    def check(self, user_id: int) -> None:
        """Check if a user has exceeded the rate limit.

        Records the current timestamp for the user. If the user has sent
        more than max_messages within the window, raises ValidationError.

        Args:
            user_id: Telegram user ID.

        Raises:
            ValidationError: If the user has exceeded the rate limit.
        """
        now = time.monotonic()
        timestamps = self._timestamps.get(user_id, [])

        timestamps = [t for t in timestamps if now - t < self.window_seconds]

        if len(timestamps) >= self.max_messages:
            raise ValidationError(
                "You're sending messages too quickly. "
                "Please wait a moment before trying again."
            )

        timestamps.append(now)
        self._timestamps[user_id] = timestamps


def _utc_today() -> date:
    """Return the date it is now in UTC.

    A function of its own so a test can move the clock, and so the one place
    the wall clock is read is named. RateLimiter uses time.monotonic() to be
    immune to the clock changing at all, but a monotonic counter has no
    notion of a date, and a daily allowance needs one.
    """
    return datetime.now(timezone.utc).date()


@dataclass
class _DayUsage:
    """What one chat has spent, and the day it spent it on.

    Attributes:
        day: The UTC date these counts belong to.
        counts: Messages sent so far today, one entry per kind.
    """

    day: date
    counts: dict[str, int]


class DailyQuota:
    """Cap how much a chat may spend in a day, per kind of message.

    The companion to RateLimiter at the other timescale: that one caps how
    fast a user spends, this one how much. Without it a chat can sit just
    inside the per-minute window all day and run the OpenAI bill up without
    limit.

    Counted per chat, and in memory only, the same way RateLimiter is - a
    restart hands every chat its day back. That is the accepted trade for
    keeping this a plain object with no file to write, lock, or repair.

    Attributes:
        limits: How many messages of each kind a chat may send per day.
            Deliberately a plain mutable attribute: the operator moves these
            from their chat at runtime, the way they move the log handler's
            push_level.
    """

    def __init__(self, text_limit: int, photo_limit: int) -> None:
        self.limits = {KIND_TEXT: text_limit, KIND_PHOTO: photo_limit}
        self._used: dict[int, _DayUsage] = {}

    def check(self, chat_id: int, kind: str) -> None:
        """Say whether a chat may still send this kind of message today.

        Spends nothing, which is the whole reason it is separate from
        `record`. A photo is checked twice - before the sender is offered
        the choice, and again before it is read - and charged once, after
        the model has actually answered. So a failed download, an OpenAI
        outage, or a declined offer costs a user nothing, and an allowance
        of 5 photos cannot be burnt by five failures.

        Args:
            chat_id: The chat the message arrived in.
            kind: KIND_TEXT or KIND_PHOTO.

        Raises:
            ValidationError: If this kind's allowance is already spent.
        """
        if self.usage(chat_id)[kind] >= self.limits[kind]:
            raise ValidationError(QUOTA_EXHAUSTED[kind].format(limit=self.limits[kind]))

    def record(self, chat_id: int, kind: str) -> None:
        """Charge one delivered message to a chat's allowance.

        Called once the work is done and paid for, so what the counters hold
        is what the bot actually spent rather than what it was asked for.

        Args:
            chat_id: The chat the message arrived in.
            kind: KIND_TEXT or KIND_PHOTO.
        """
        self._counts_today(chat_id)[kind] += 1

    def usage(self, chat_id: int) -> dict[str, int]:
        """Report what a chat has spent today, without spending anything.

        Args:
            chat_id: The chat being asked about.

        Returns:
            One count per kind; all zero for a chat that has not spoken
            today, including one whose counters are left over from
            yesterday.
        """
        usage = self._used.get(chat_id)
        if usage is None or usage.day != _utc_today():
            return dict.fromkeys(QUOTA_KINDS, 0)
        return dict(usage.counts)

    def spent_today(self) -> dict[str, int]:
        """Total what every chat together has spent today.

        The operator needs this to choose the limits at all: the right
        numbers cannot be guessed from an empty bot, only read off what
        people actually send.

        Returns:
            One total per kind, over the chats that have spoken today.
        """
        totals = dict.fromkeys(QUOTA_KINDS, 0)
        for usage in self._today_only():
            for kind, count in usage.counts.items():
                totals[kind] += count
        return totals

    def active_chats(self) -> int:
        """Count the chats the bot has answered today.

        Returns:
            How many chats have had at least one message delivered.
        """
        return sum(1 for _ in self._today_only())

    def reset(self) -> None:
        """Give every chat its day back, for the operator's panel button."""
        self._used.clear()

    def _today_only(self) -> Iterator[_DayUsage]:
        """Walk the entries that belong to today, skipping yesterday's."""
        today = _utc_today()
        return (usage for usage in self._used.values() if usage.day == today)

    def _counts_today(self, chat_id: int) -> dict[str, int]:
        """Return the counters to charge, starting a fresh day if needed.

        Only `record` calls this, so a chat that is turned away never gets
        an entry and never counts towards `active_chats`. Yesterday's entry
        is replaced where it is found rather than swept up on a timer, so a
        chat that stops talking costs nothing to forget.

        Args:
            chat_id: The chat the message arrived in.

        Returns:
            The live counts for today, to be mutated by the caller.
        """
        today = _utc_today()
        usage = self._used.get(chat_id)
        if usage is None or usage.day != today:
            usage = _DayUsage(day=today, counts=dict.fromkeys(QUOTA_KINDS, 0))
            self._used[chat_id] = usage
        return usage.counts
