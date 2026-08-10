from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError, APITimeoutError, RateLimitError

from src.bot.handlers import (
    ERROR_CONNECTION,
    ERROR_GENERIC,
    ERROR_RATE_LIMIT,
    ERROR_TIMEOUT,
    handle_message,
    help_command,
    pref_command,
    start_command,
)
from src.bot.middleware import RateLimiter


async def test_start_command_sends_welcome_message(mock_update, mock_context):
    await start_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Welcome" in reply_text
    assert "Immigrant Assistance Bot" in reply_text


async def test_start_command_mentions_key_features(mock_update, mock_context):
    await start_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Serbia" in reply_text
    assert "Translation" in reply_text or "translation" in reply_text


async def test_help_command_sends_help_message(mock_update, mock_context):
    await help_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "/start" in reply_text
    assert "/help" in reply_text


async def test_help_command_includes_usage_examples(mock_update, mock_context):
    await help_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Example" in reply_text or "example" in reply_text


async def test_help_command_lists_pref_command(mock_update, mock_context):
    await help_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "/pref" in reply_text


@patch("src.bot.handlers.get_preferences", return_value=["Reply in Cyrillic."])
async def test_pref_command_no_args_shows_saved_preferences(
    mock_get, mock_update, mock_context
):
    mock_context.args = []

    await pref_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Reply in Cyrillic." in reply_text


@patch("src.bot.handlers.get_preferences", return_value=[])
async def test_pref_command_no_args_and_none_saved_shows_usage(
    mock_get, mock_update, mock_context
):
    mock_context.args = []

    await pref_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "no saved preferences" in reply_text
    assert "/pref add" in reply_text


@patch("src.bot.handlers.add_preference", return_value=["Write in Cyrillic"])
async def test_pref_command_add_saves_the_rule(mock_add, mock_update, mock_context):
    mock_context.args = ["add", "Write", "in", "Cyrillic"]

    await pref_command(mock_update, mock_context)

    mock_add.assert_called_once()
    saved_rule = mock_add.call_args.args[2]
    assert saved_rule == "Write in Cyrillic"
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Saved" in reply_text


@patch("src.bot.handlers.add_preference")
async def test_pref_command_add_without_a_rule_asks_for_one(
    mock_add, mock_update, mock_context
):
    mock_context.args = ["add"]

    await pref_command(mock_update, mock_context)

    mock_add.assert_not_called()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "rule" in reply_text.lower()


@patch("src.bot.handlers.clear_preferences")
async def test_pref_command_clear_removes_preferences(
    mock_clear, mock_update, mock_context
):
    mock_context.args = ["clear"]

    await pref_command(mock_update, mock_context)

    mock_clear.assert_called_once()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Cleared" in reply_text


async def test_pref_command_unknown_action_shows_usage(mock_update, mock_context):
    mock_context.args = ["wobble"]

    await pref_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "/pref add" in reply_text


@patch("src.bot.handlers.remove_preference", return_value=["Keep it short."])
@patch("src.bot.handlers.get_preferences", return_value=["A rule.", "Keep it short."])
async def test_pref_command_remove_deletes_by_number(
    mock_get, mock_remove, mock_update, mock_context
):
    mock_context.args = ["remove", "1"]

    await pref_command(mock_update, mock_context)

    # The 1-based number is converted to a 0-based index for the helper.
    assert mock_remove.call_args.args[2] == 0
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Removed" in reply_text


@patch("src.bot.handlers.remove_preference")
@patch("src.bot.handlers.get_preferences", return_value=["A rule."])
async def test_pref_command_remove_without_a_number_asks(
    mock_get, mock_remove, mock_update, mock_context
):
    mock_context.args = ["remove"]

    await pref_command(mock_update, mock_context)

    mock_remove.assert_not_called()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "number" in reply_text.lower()


@patch("src.bot.handlers.remove_preference")
@patch("src.bot.handlers.get_preferences", return_value=["A rule."])
async def test_pref_command_remove_out_of_range_reports_it(
    mock_get, mock_remove, mock_update, mock_context
):
    mock_context.args = ["remove", "9"]

    await pref_command(mock_update, mock_context)

    mock_remove.assert_not_called()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "no preference number 9" in reply_text.lower()


@patch(
    "src.bot.handlers.tidy_preferences",
    new=AsyncMock(return_value=["Merged rule."]),
)
async def test_pref_command_tidy_merges_and_shows_result(mock_update, mock_context):
    mock_context.args = ["tidy"]
    mock_context.bot_data["preference_tidier"] = MagicMock()

    await pref_command(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "Tidied" in reply_text
    assert "Merged rule." in reply_text


async def test_handle_message_sends_orchestrator_response(mock_update, mock_context):
    mock_update.message.text = "How do I get a work permit?"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(
        "Test answer from orchestrator."
    )


async def test_handle_message_passes_user_text_to_orchestrator(
    mock_update, mock_context, mock_orchestrator
):
    mock_update.message.text = "What is a White Card?"

    await handle_message(mock_update, mock_context)

    mock_orchestrator.ainvoke.assert_called_once()
    sent = mock_orchestrator.ainvoke.call_args.args[0]["messages"][-1].content
    assert sent == "What is a White Card?"


async def test_handle_message_uses_chat_id_as_thread(
    mock_update, mock_context, mock_orchestrator
):
    mock_update.message.text = "What is a White Card?"

    await handle_message(mock_update, mock_context)

    config = mock_orchestrator.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["thread_id"] == str(mock_update.message.chat_id)


async def test_handle_message_rejects_empty_text(
    mock_update, mock_context, mock_orchestrator
):
    mock_update.message.text = "   "

    await handle_message(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "empty" in reply_text
    mock_orchestrator.ainvoke.assert_not_called()


async def test_handle_message_rejects_too_long_text(
    mock_update, mock_context, mock_orchestrator
):
    mock_update.message.text = "a" * 1500

    await handle_message(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "too long" in reply_text
    mock_orchestrator.ainvoke.assert_not_called()


async def test_handle_message_replies_timeout_on_llm_timeout(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APITimeoutError(request=None))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_TIMEOUT)


async def test_handle_message_replies_connection_error_on_network_failure(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APIConnectionError(request=None))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_CONNECTION)


async def test_handle_message_replies_rate_limit_on_openai_limit(
    mock_update, mock_context, mock_orchestrator
):
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_RATE_LIMIT)


async def test_handle_message_replies_generic_error_on_unexpected_failure(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=RuntimeError("something unexpected")
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_GENERIC)


async def test_handle_message_rejects_when_rate_limited(mock_update, mock_orchestrator):
    rate_limiter = RateLimiter(max_messages=1, window_seconds=60)
    context = MagicMock()
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": rate_limiter,
    }
    mock_update.message.text = "first question"

    await handle_message(mock_update, context)
    mock_update.message.reply_text.reset_mock()

    mock_update.message.text = "second question"
    await handle_message(mock_update, context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "too quickly" in reply_text
    assert mock_orchestrator.ainvoke.call_count == 1


async def test_handle_message_strips_whitespace_before_sending(
    mock_update, mock_context, mock_orchestrator
):
    mock_update.message.text = "   What is a White Card?   "

    await handle_message(mock_update, mock_context)

    mock_orchestrator.ainvoke.assert_called_once()
    sent = mock_orchestrator.ainvoke.call_args.args[0]["messages"][-1].content
    assert sent == "What is a White Card?"
