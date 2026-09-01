from datetime import date
from unittest.mock import patch

import pytest

from src.bot.middleware import (
    KIND_PHOTO,
    KIND_TEXT,
    LARGE_IMAGE_BYTES,
    MAX_IMAGE_BYTES,
    MAX_MESSAGE_LENGTH,
    QUOTA_EXHAUSTED,
    DailyQuota,
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


def test_daily_quota_check_is_pure_and_never_exhausts_the_allowance():
    # check() spends nothing; only record() does. Calling it repeatedly must
    # never burn through the allowance on its own.
    quota = DailyQuota(text_limit=1, photo_limit=1)

    for _ in range(5):
        quota.check(chat_id=1, kind=KIND_TEXT)


def test_daily_quota_check_raises_once_the_allowance_is_recorded():
    quota = DailyQuota(text_limit=2, photo_limit=2)
    quota.record(chat_id=1, kind=KIND_TEXT)
    quota.record(chat_id=1, kind=KIND_TEXT)

    with pytest.raises(ValidationError):
        quota.check(chat_id=1, kind=KIND_TEXT)


def test_daily_quota_check_raises_the_quota_exhausted_message_for_the_kind():
    quota = DailyQuota(text_limit=1, photo_limit=1)
    quota.record(chat_id=1, kind=KIND_PHOTO)

    with pytest.raises(ValidationError) as exc_info:
        quota.check(chat_id=1, kind=KIND_PHOTO)

    assert exc_info.value.user_message == QUOTA_EXHAUSTED[KIND_PHOTO].format(limit=1)


def test_daily_quota_spending_every_photo_leaves_text_untouched():
    quota = DailyQuota(text_limit=5, photo_limit=1)
    quota.record(chat_id=1, kind=KIND_PHOTO)

    quota.check(chat_id=1, kind=KIND_TEXT)  # does not raise


def test_daily_quota_spending_every_text_leaves_photo_untouched():
    quota = DailyQuota(text_limit=1, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)

    quota.check(chat_id=1, kind=KIND_PHOTO)  # does not raise


def test_daily_quota_a_limit_of_zero_refuses_the_first_check():
    # /limits text 0 is explicitly supported, so a chat may be switched off.
    quota = DailyQuota(text_limit=0, photo_limit=5)

    with pytest.raises(ValidationError):
        quota.check(chat_id=1, kind=KIND_TEXT)


@patch("src.bot.middleware._utc_today")
def test_daily_quota_a_new_day_clears_the_counts(mock_today):
    mock_today.return_value = date(2024, 1, 1)
    quota = DailyQuota(text_limit=1, photo_limit=1)
    quota.record(chat_id=1, kind=KIND_TEXT)

    mock_today.return_value = date(2024, 1, 2)

    quota.check(chat_id=1, kind=KIND_TEXT)  # does not raise


def test_daily_quota_usage_reports_todays_spend():
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)
    quota.record(chat_id=1, kind=KIND_TEXT)

    assert quota.usage(chat_id=1)[KIND_TEXT] == 2


@patch("src.bot.middleware._utc_today")
def test_daily_quota_usage_returns_zeros_for_a_chat_whose_only_entry_is_from_yesterday(
    mock_today,
):
    mock_today.return_value = date(2024, 1, 1)
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)

    mock_today.return_value = date(2024, 1, 2)

    assert quota.usage(chat_id=1) == {KIND_TEXT: 0, KIND_PHOTO: 0}


def test_daily_quota_spent_today_totals_across_chats():
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)
    quota.record(chat_id=2, kind=KIND_TEXT)

    assert quota.spent_today()[KIND_TEXT] == 2


@patch("src.bot.middleware._utc_today")
def test_daily_quota_spent_today_ignores_yesterdays_entries(mock_today):
    mock_today.return_value = date(2024, 1, 1)
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)

    mock_today.return_value = date(2024, 1, 2)
    quota.record(chat_id=2, kind=KIND_TEXT)

    assert quota.spent_today()[KIND_TEXT] == 1


def test_daily_quota_active_chats_counts_chats_with_a_recorded_message_today():
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)
    quota.record(chat_id=2, kind=KIND_PHOTO)

    assert quota.active_chats() == 2


def test_daily_quota_a_chat_that_was_only_checked_does_not_appear_in_active_chats():
    # check() must not create an entry, or a chat that was only ever turned
    # away would count as one the bot actually answered.
    quota = DailyQuota(text_limit=0, photo_limit=5)

    with pytest.raises(ValidationError):
        quota.check(chat_id=1, kind=KIND_TEXT)

    assert quota.active_chats() == 0


def test_daily_quota_reset_zeroes_every_chat():
    quota = DailyQuota(text_limit=5, photo_limit=5)
    quota.record(chat_id=1, kind=KIND_TEXT)
    quota.record(chat_id=2, kind=KIND_PHOTO)

    quota.reset()

    assert quota.spent_today() == {KIND_TEXT: 0, KIND_PHOTO: 0}
