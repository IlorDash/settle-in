"""Middleware utilities for input validation and rate limiting.

These functions act as guards that run before the main handler logic.
They protect against invalid input and excessive usage.
"""

import time

MAX_MESSAGE_LENGTH = 1000
RATE_LIMIT_MESSAGES = 5
RATE_LIMIT_WINDOW_SECONDS = 60

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
