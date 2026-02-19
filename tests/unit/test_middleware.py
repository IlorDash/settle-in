from unittest.mock import patch

import pytest

from src.bot.middleware import (
    MAX_MESSAGE_LENGTH,
    RateLimiter,
    ValidationError,
    validate_message_text,
)


def test_validate_message_text_returns_stripped_text():
    result = validate_message_text("  Hello world  ")

    assert result == "Hello world"


def test_validate_message_text_raises_on_none():
    with pytest.raises(ValidationError) as exc_info:
        validate_message_text(None)

    assert "empty" in exc_info.value.user_message


def test_validate_message_text_raises_on_empty_string():
    with pytest.raises(ValidationError) as exc_info:
        validate_message_text("")

    assert "empty" in exc_info.value.user_message


def test_validate_message_text_raises_on_whitespace_only():
    with pytest.raises(ValidationError) as exc_info:
        validate_message_text("   \n\t  ")

    assert "empty" in exc_info.value.user_message


def test_validate_message_text_raises_on_too_long_message():
    long_text = "a" * (MAX_MESSAGE_LENGTH + 1)

    with pytest.raises(ValidationError) as exc_info:
        validate_message_text(long_text)

    assert "too long" in exc_info.value.user_message


def test_validate_message_text_accepts_max_length():
    text = "a" * MAX_MESSAGE_LENGTH

    result = validate_message_text(text)

    assert len(result) == MAX_MESSAGE_LENGTH


def test_rate_limiter_allows_messages_under_limit():
    limiter = RateLimiter(max_messages=3, window_seconds=60)
    user_id = 123

    limiter.check(user_id)
    limiter.check(user_id)
    limiter.check(user_id)


def test_rate_limiter_blocks_when_limit_exceeded():
    limiter = RateLimiter(max_messages=2, window_seconds=60)
    user_id = 123

    limiter.check(user_id)
    limiter.check(user_id)

    with pytest.raises(ValidationError) as exc_info:
        limiter.check(user_id)

    assert "too quickly" in exc_info.value.user_message


def test_rate_limiter_tracks_users_independently():
    limiter = RateLimiter(max_messages=1, window_seconds=60)

    limiter.check(user_id=100)

    with pytest.raises(ValidationError):
        limiter.check(user_id=100)

    limiter.check(user_id=200)


@patch("src.bot.middleware.time.monotonic")
def test_rate_limiter_resets_after_window_expires(mock_monotonic):
    limiter = RateLimiter(max_messages=1, window_seconds=60)
    user_id = 123

    mock_monotonic.return_value = 0.0
    limiter.check(user_id)

    mock_monotonic.return_value = 61.0
    limiter.check(user_id)
