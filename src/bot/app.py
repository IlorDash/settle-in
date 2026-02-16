import logging

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from src.bot.handlers import handle_message, help_command, start_command
from src.config import settings

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_application():
    """Build the bot application and register all handlers.

    Returns:
        Configured Application ready to be started.
    """
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot handlers registered successfully.")
    return app


def main() -> None:
    """Start the bot in the mode specified by BOT_MODE env variable.

    Raises:
        NotImplementedError: If bot_mode is "webhook" (not yet supported).
    """
    logger.info("Starting bot in %s mode...", settings.bot_mode)

    app = create_application()

    if settings.bot_mode == "webhook":
        raise NotImplementedError("Webhook mode will be added in Phase 5.")

    logger.info("Bot is polling for updates... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
