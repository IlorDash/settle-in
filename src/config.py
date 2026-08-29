import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# What one chat may spend in a day before it is turned away. The operator
# moves both from their chat with /limits; these are where a restart puts
# them back, so a number that proves right belongs in the environment.
DEFAULT_DAILY_TEXT_LIMIT = 30
# Smaller on purpose: the vision model is billed by the pixels it is handed,
# so a photo costs several times what a text turn costs.
DEFAULT_DAILY_PHOTO_LIMIT = 5


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables.

    Attributes:
        telegram_bot_token: Bot token from BotFather.
        openai_api_key: OpenAI API key for LLM and embedding calls.
        chroma_persist_dir: Path where ChromaDB persists data to disk.
        feedback_path: JSONL file collecting thumbs up/down verdicts.
        checkpoint_path: SQLite file holding per-chat history and preferences.
        bot_mode: "polling" for local dev, "webhook" for production.
        webhook_url: Public HTTPS URL for webhook mode.
        port: Port number for the webhook server (default 8443).
        admin_chat_id: Chat that receives operator alerts; empty turns them off.
        daily_text_limit: Text messages one chat may send per day.
        daily_photo_limit: Photos one chat may send per day; smaller, because
            a photo costs several times what a text turn costs.
        announcement_channel: Public channel the bot posts announcements to,
            as its @username; empty turns announcements off.
    """

    telegram_bot_token: str
    openai_api_key: str
    chroma_persist_dir: str
    feedback_path: str
    checkpoint_path: str
    bot_mode: str
    webhook_url: str
    port: int
    admin_chat_id: str
    daily_text_limit: int
    daily_photo_limit: int
    announcement_channel: str


def load_settings() -> Settings:
    """Create a Settings instance from environment variables.

    Returns:
        Populated Settings object.

    Raises:
        ValueError: If TELEGRAM_BOT_TOKEN or OPENAI_API_KEY is not set.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Create a .env file with your bot token from BotFather."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add your OpenAI API key to the .env file."
        )

    return Settings(
        telegram_bot_token=token,
        openai_api_key=openai_api_key,
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        feedback_path=os.getenv("FEEDBACK_PATH", "./data/feedback.jsonl"),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", "./data/checkpoints.sqlite"),
        bot_mode=os.getenv("BOT_MODE", "polling"),
        webhook_url=os.getenv("WEBHOOK_URL", ""),
        port=int(os.getenv("PORT", "8443")),
        admin_chat_id=os.getenv("ADMIN_CHAT_ID", ""),
        daily_text_limit=int(
            os.getenv("DAILY_TEXT_LIMIT", str(DEFAULT_DAILY_TEXT_LIMIT))
        ),
        daily_photo_limit=int(
            os.getenv("DAILY_PHOTO_LIMIT", str(DEFAULT_DAILY_PHOTO_LIMIT))
        ),
        announcement_channel=os.getenv("ANNOUNCEMENT_CHANNEL", ""),
    )


settings = load_settings()
