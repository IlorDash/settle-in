"""Middleware utilities for input validation and rate limiting.

These functions act as guards that run before the main handler logic.
They protect against invalid input and excessive usage.
"""

import time

MAX_MESSAGE_LENGTH = 1000
RATE_LIMIT_MESSAGES = 5
RATE_LIMIT_WINDOW_SECONDS = 60


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
