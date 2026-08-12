from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import Chat, InaccessibleMessage
from telegram.error import BadRequest, TelegramError

from src.bot.handlers import (
    ERROR_CONNECTION,
    ERROR_GENERIC,
    ERROR_RATE_LIMIT,
    ERROR_TIMEOUT,
    FEEDBACK_LOST,
    FEEDBACK_THANKS,
    error_handler,
    feedback_callback,
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

    mock_update.message.reply_text.assert_called_once()
    reply_text = mock_update.message.reply_text.call_args[0][0]
    assert reply_text == "Test answer from orchestrator."


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


async def test_handle_message_attaches_feedback_buttons(mock_update, mock_context):
    # A knowledge answer is rateable, so it must carry the two buttons.
    mock_update.message.text = "What is a White Card?"

    await handle_message(mock_update, mock_context)

    markup = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    assert len(markup.inline_keyboard[0]) == 2


async def test_handle_message_replies_to_the_question(mock_update, mock_context):
    # The reply link is what lets a later tap find the original question,
    # so losing it would silently break feedback recording.
    mock_update.message.text = "What is a White Card?"

    await handle_message(mock_update, mock_context)

    reply_to = mock_update.message.reply_text.call_args.kwargs["reply_to_message_id"]
    assert reply_to == mock_update.message.message_id


async def test_handle_message_omits_buttons_for_out_of_scope(
    mock_update, mock_context, mock_orchestrator
):
    # Rating a rejection teaches the classifier nothing, so no buttons.
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "out_of_scope", "agent_response": "Out of scope."}
    )
    mock_update.message.text = "tell me a joke"

    await handle_message(mock_update, mock_context)

    assert mock_update.message.reply_text.call_args.kwargs["reply_markup"] is None


async def test_handle_message_error_reply_has_no_buttons(
    mock_update, mock_context, mock_orchestrator
):
    # Error replies go through a different reply_text call with no markup.
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APITimeoutError(request=None))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    assert mock_update.message.reply_text.call_args.kwargs == {}


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_records_the_verdict(mock_record, mock_callback_update):
    await feedback_callback(mock_callback_update, MagicMock())

    assert mock_record.call_args.args == (
        "What is a White Card?",
        "knowledge_question",
        "up",
    )


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_thanks_the_user(mock_record, mock_callback_update):
    await feedback_callback(mock_callback_update, MagicMock())

    mock_callback_update.callback_query.answer.assert_awaited_once_with(FEEDBACK_THANKS)


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_removes_the_buttons(mock_record, mock_callback_update):
    # Clearing the keyboard is what stops the same answer being rated twice.
    await feedback_callback(mock_callback_update, MagicMock())

    query = mock_callback_update.callback_query
    query.edit_message_reply_markup.assert_awaited_once_with(None)


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_ignores_unknown_payload(
    mock_record, mock_callback_update
):
    # A button from an older deploy must be shrugged off, not unpacked.
    mock_callback_update.callback_query.data = "something:else"

    await feedback_callback(mock_callback_update, MagicMock())

    mock_record.assert_not_called()


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_rejects_an_unknown_verdict(
    mock_record, mock_callback_update
):
    # The shape is right but the verdict is not one we issue.
    mock_callback_update.callback_query.data = "fb:sideways:translation"

    await feedback_callback(mock_callback_update, MagicMock())

    mock_record.assert_not_called()


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_reports_a_missing_question(
    mock_record, mock_callback_update
):
    # The user deleted the question, so the reply chain no longer reaches it.
    mock_callback_update.callback_query.message.reply_to_message = None

    await feedback_callback(mock_callback_update, MagicMock())

    mock_callback_update.callback_query.answer.assert_awaited_once_with(FEEDBACK_LOST)


@patch("src.bot.handlers.record_feedback", side_effect=OSError("read-only file system"))
async def test_feedback_callback_still_thanks_when_saving_fails(
    mock_record, mock_callback_update
):
    # A full or read-only disk is our problem, not something the user sees.
    await feedback_callback(mock_callback_update, MagicMock())

    mock_callback_update.callback_query.answer.assert_awaited_once_with(FEEDBACK_THANKS)


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_survives_a_stale_keyboard(
    mock_record, mock_callback_update
):
    # Telegram refuses the edit on a second tap; the vote is already saved.
    query = mock_callback_update.callback_query
    query.edit_message_reply_markup = AsyncMock(side_effect=BadRequest("not modified"))

    await feedback_callback(mock_callback_update, MagicMock())

    mock_record.assert_called_once()


@patch("src.bot.handlers.record_feedback")
async def test_feedback_callback_handles_an_inaccessible_answer(
    mock_record, mock_callback_update
):
    # Past ~48 hours Telegram sends an InaccessibleMessage, which carries no
    # reply_to_message at all. Reading it used to raise AttributeError before
    # the query was ever answered, leaving the button spinning.
    inaccessible = InaccessibleMessage(chat=Chat(id=1, type="private"), message_id=5)
    mock_callback_update.callback_query.message = inaccessible

    await feedback_callback(mock_callback_update, MagicMock())

    mock_callback_update.callback_query.answer.assert_awaited_once_with(FEEDBACK_LOST)


async def test_error_handler_apologises_to_the_user(mock_update):
    # Without this handler an unexpected crash leaves the user with silence.
    context = MagicMock()
    context.error = RuntimeError("boom")
    mock_update.effective_message = mock_update.message

    await error_handler(mock_update, context)

    mock_update.message.reply_text.assert_awaited_once_with(ERROR_GENERIC)


async def test_error_handler_survives_an_update_less_failure():
    # A background failure has no update and nobody to reply to, so the
    # handler must log and return rather than raise a second error.
    context = MagicMock()
    context.error = RuntimeError("boom")

    await error_handler(None, context)


async def test_error_handler_survives_an_undeliverable_apology(mock_update):
    # The user may have blocked the bot; the log is what matters by then.
    context = MagicMock()
    context.error = RuntimeError("boom")
    mock_update.effective_message = mock_update.message
    mock_update.message.reply_text = AsyncMock(side_effect=TelegramError("blocked"))

    await error_handler(mock_update, context)


async def test_handle_message_attaches_buttons_to_translations(
    mock_update, mock_context, mock_orchestrator
):
    # The other rateable intent, so the keyboard is not knowledge-only.
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "translation", "agent_response": "papagaj"}
    )
    mock_update.message.text = "Переведи попугай на сербский"

    await handle_message(mock_update, mock_context)

    markup = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].callback_data == "fb:up:translation"
