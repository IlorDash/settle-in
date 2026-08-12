import logging

import src.bot.app  # noqa: F401  (imported for its logging configuration)


def test_httpx_logging_is_pinned_to_warning_to_keep_the_token_out_of_logs():
    # Telegram puts the bot token in the request URL, and httpx logs every
    # URL at INFO. Left on, that copies the token into the deployment logs
    # on every call, so anyone with log access owns the bot.
    #
    # This asserts the logger's own level rather than isEnabledFor(): pytest
    # attaches its own root handlers, which makes logging.basicConfig a no-op
    # and leaves the root logger at WARNING. Every logger would then look
    # "not enabled for INFO" during tests, so isEnabledFor would pass just as
    # happily with this protection removed.
    assert logging.getLogger("httpx").level == logging.WARNING
