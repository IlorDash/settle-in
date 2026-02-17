from unittest.mock import AsyncMock, MagicMock

from openai import APIConnectionError, APITimeoutError, RateLimitError

from src.bot.handlers import (
    ERROR_CONNECTION,
    ERROR_GENERIC,
    ERROR_RATE_LIMIT,
    ERROR_TIMEOUT,
    handle_message,
    help_command,
    start_command,
)


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

    mock_orchestrator.ainvoke.assert_called_once_with(
        {"user_message": "What is a White Card?"}
    )


async def test_handle_message_replies_error_on_orchestrator_failure(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert "something went wrong" in reply_text


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
