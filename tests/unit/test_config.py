from unittest.mock import patch

import pytest

from src.config import (
    DEFAULT_DAILY_PHOTO_LIMIT,
    DEFAULT_DAILY_TEXT_LIMIT,
    load_settings,
)


@patch.dict(
    "os.environ", {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"}
)
def test_load_settings_returns_settings_with_env_vars():
    settings = load_settings()

    assert settings.telegram_bot_token == "test-token"
    assert settings.openai_api_key == "test-key"


@patch.dict(
    "os.environ", {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"}
)
def test_load_settings_uses_default_values():
    settings = load_settings()

    assert settings.chroma_persist_dir == "./data/chroma_db"
    assert settings.feedback_path == "./data/feedback.jsonl"
    assert settings.checkpoint_path == "./data/checkpoints.sqlite"
    assert settings.bot_mode == "polling"
    assert settings.webhook_url == ""


@patch.dict(
    "os.environ",
    {
        "TELEGRAM_BOT_TOKEN": "t",
        "OPENAI_API_KEY": "k",
        "CHROMA_PERSIST_DIR": "/custom/path",
        "FEEDBACK_PATH": "/mnt/volume/feedback.jsonl",
        "CHECKPOINT_PATH": "/mnt/volume/checkpoints.sqlite",
        "BOT_MODE": "webhook",
        "WEBHOOK_URL": "https://example.com/webhook",
    },
)
def test_load_settings_reads_custom_env_values():
    settings = load_settings()

    assert settings.chroma_persist_dir == "/custom/path"
    assert settings.bot_mode == "webhook"
    assert settings.webhook_url == "https://example.com/webhook"


@patch.dict(
    "os.environ",
    {
        "TELEGRAM_BOT_TOKEN": "t",
        "OPENAI_API_KEY": "k",
        "FEEDBACK_PATH": "/mnt/volume/feedback.jsonl",
        "CHECKPOINT_PATH": "/mnt/volume/checkpoints.sqlite",
    },
)
def test_load_settings_reads_writable_paths_from_env():
    # These two exist so a deployment can move everything the bot writes onto
    # a mounted volume; hardcoding them would make the data unsavable.
    settings = load_settings()

    assert settings.feedback_path == "/mnt/volume/feedback.jsonl"
    assert settings.checkpoint_path == "/mnt/volume/checkpoints.sqlite"


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
def test_load_settings_raises_when_telegram_token_missing():
    with pytest.raises(ValueError) as exc_info:
        load_settings()

    assert "TELEGRAM_BOT_TOKEN" in str(exc_info.value)


@patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True)
def test_load_settings_raises_when_openai_key_missing():
    with pytest.raises(ValueError) as exc_info:
        load_settings()

    assert "OPENAI_API_KEY" in str(exc_info.value)


@patch.dict(
    "os.environ",
    {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"},
    clear=True,
)
def test_load_settings_admin_chat_id_defaults_to_empty():
    settings = load_settings()

    assert settings.admin_chat_id == ""


@patch.dict(
    "os.environ",
    {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "ADMIN_CHAT_ID": "123456",
    },
)
def test_load_settings_reads_admin_chat_id_from_env():
    settings = load_settings()

    assert settings.admin_chat_id == "123456"


@patch.dict(
    "os.environ",
    {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"},
    clear=True,
)
def test_load_settings_announcement_channel_defaults_to_empty():
    settings = load_settings()

    assert settings.announcement_channel == ""


@patch.dict(
    "os.environ",
    {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"},
    clear=True,
)
def test_load_settings_daily_text_limit_defaults_to_the_default_constant():
    settings = load_settings()

    assert settings.daily_text_limit == DEFAULT_DAILY_TEXT_LIMIT


@patch.dict(
    "os.environ",
    {"TELEGRAM_BOT_TOKEN": "test-token", "OPENAI_API_KEY": "test-key"},
    clear=True,
)
def test_load_settings_daily_photo_limit_defaults_to_the_default_constant():
    settings = load_settings()

    assert settings.daily_photo_limit == DEFAULT_DAILY_PHOTO_LIMIT


@patch.dict(
    "os.environ",
    {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "DAILY_TEXT_LIMIT": "50",
    },
)
def test_load_settings_reads_daily_text_limit_from_env_as_an_int():
    settings = load_settings()

    assert settings.daily_text_limit == 50


@patch.dict(
    "os.environ",
    {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "DAILY_PHOTO_LIMIT": "10",
    },
)
def test_load_settings_reads_daily_photo_limit_from_env_as_an_int():
    settings = load_settings()

    assert settings.daily_photo_limit == 10
