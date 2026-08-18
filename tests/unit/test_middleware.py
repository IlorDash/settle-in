from unittest.mock import patch

import pytest

from src.bot.middleware import (
    LARGE_IMAGE_BYTES,
    MAX_IMAGE_BYTES,
    MAX_MESSAGE_LENGTH,
    RateLimiter,
    ValidationError,
    is_large_upload,
    validate_image_upload,
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


def test_validate_image_upload_accepts_an_ordinary_photo():
    validate_image_upload(120_000, "image/jpeg")


def test_validate_image_upload_rejects_an_unsupported_type():
    # An iPhone HEIC sent as a file: the vision API would refuse it anyway,
    # so the user gets a clear answer instead of a generic failure.
    with pytest.raises(ValidationError) as exc_info:
        validate_image_upload(120_000, "image/heic")

    assert "JPEG" in exc_info.value.user_message


def test_validate_image_upload_rejects_an_oversized_image():
    with pytest.raises(ValidationError) as exc_info:
        validate_image_upload(MAX_IMAGE_BYTES + 1, "image/jpeg")

    assert "too large" in exc_info.value.user_message


def test_validate_image_upload_accepts_an_image_at_the_limit():
    validate_image_upload(MAX_IMAGE_BYTES, "image/jpeg")


def test_validate_image_upload_allows_an_unreported_size():
    # Telegram does not always send file_size; the download limit still caps
    # what can arrive, so an unknown size must not block a valid photo.
    validate_image_upload(None, "image/jpeg")


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


def test_is_large_upload_flags_a_full_size_phone_photo():
    # A 3 MB document costs several times the compressed copy Telegram makes
    # when the same picture is sent as a photo.
    assert is_large_upload(3 * 1024 * 1024) is True


def test_is_large_upload_ignores_an_ordinary_telegram_photo():
    assert is_large_upload(120_000) is False


def test_is_large_upload_ignores_an_unreported_size():
    # Telegram does not always send file_size; never warn on a guess.
    assert is_large_upload(None) is False


def test_is_large_upload_leaves_the_threshold_itself_unflagged():
    assert is_large_upload(LARGE_IMAGE_BYTES) is False
