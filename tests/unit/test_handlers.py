import logging
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from telegram import Chat, InaccessibleMessage
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError

from src.bot.admin import TelegramLogHandler
from src.bot.handlers import (
    ERROR_CONNECTION,
    ERROR_GENERIC,
    ERROR_NEEDS_OPERATOR,
    ERROR_OUTAGE,
    ERROR_RATE_LIMIT,
    ERROR_TIMEOUT,
    ERROR_TOO_LONG,
    EXPORT_EMPTY,
    EXPORT_LIMIT,
    FEATURE_EXPORT,
    FEATURE_RESET,
    FEATURE_UNAVAILABLE,
    FEEDBACK_LOST,
    FEEDBACK_THANKS,
    LARGE_FILE_CANCELLED,
    LIMITS_BAD_ARGS,
    LIMITS_PERSONAL,
    LOGLEVEL_SET,
    LOGLEVEL_USAGE,
    LOGS_EMPTY,
    RESET_COUNTERS_LABEL,
    RESET_DONE,
    RESET_EMPTY,
    TELEGRAM_MESSAGE_LIMIT,
    UNSUPPORTED_FILE,
    _admin_keyboard,
    _admin_panel_text,
    _export_limit,
    _limits_report,
    _limits_status,
    _PanelState,
    _split_for_telegram,
    _strip_markdown,
    admin_callback,
    admin_command,
    default_features,
    document_callback,
    error_handler,
    export_command,
    feedback_callback,
    handle_message,
    handle_photo,
    handle_unsupported_file,
    help_command,
    limits_command,
    loglevel_command,
    logs_command,
    pref_command,
    reset_command,
    start_command,
)
from src.bot.middleware import (
    KIND_PHOTO,
    KIND_TEXT,
    MAX_IMAGE_BYTES,
    QUOTA_EXHAUSTED,
    DailyQuota,
    RateLimiter,
)


def _api_error(cls, status_code, code=None, error_type=None):
    """Build a real openai exception the way the SDK would raise it.

    The SDK strips the "error" wrapper before constructing the exception, so
    the body is flat: {"type": ..., "code": ...}.
    """
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    body = {"type": error_type or code, "code": code} if code is not None else None
    return cls(message="mock error", response=response, body=body)


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


@patch(
    "src.bot.handlers.tidy_preferences",
    new=AsyncMock(side_effect=APITimeoutError(request=None)),
)
async def test_pref_command_tidy_replies_timeout_on_llm_timeout(
    mock_update, mock_context
):
    mock_context.args = ["tidy"]
    mock_context.bot_data["preference_tidier"] = MagicMock()

    await pref_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_TIMEOUT)


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


async def test_handle_message_replies_needs_operator_on_exhausted_quota(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(RateLimitError, 429, code="insufficient_quota")
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_NEEDS_OPERATOR)


async def test_handle_message_replies_rate_limit_on_explicit_throttle_code(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(RateLimitError, 429, code="rate_limit_exceeded")
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_RATE_LIMIT)


async def test_handle_message_replies_outage_on_internal_server_error(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(InternalServerError, 500)
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_OUTAGE)


async def test_handle_message_replies_needs_operator_on_bad_authentication(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(AuthenticationError, 401)
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_NEEDS_OPERATOR)


async def test_handle_message_replies_needs_operator_on_permission_denied(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(PermissionDeniedError, 403)
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_NEEDS_OPERATOR)


async def test_handle_message_replies_needs_operator_on_model_not_found(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(side_effect=_api_error(NotFoundError, 404))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_NEEDS_OPERATOR)


async def test_handle_message_replies_too_long_on_context_length_exceeded(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(BadRequestError, 400, code="context_length_exceeded")
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_TOO_LONG)


async def test_handle_message_replies_generic_on_other_bad_request_codes(
    mock_update, mock_context, mock_orchestrator
):
    # Guards the compound condition on branch 6: without the `code` check,
    # a nested `if` would route every 400 to ERROR_TOO_LONG.
    mock_orchestrator.ainvoke = AsyncMock(
        side_effect=_api_error(BadRequestError, 400, code="invalid_image_url")
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(ERROR_GENERIC)


async def test_handle_message_rejects_when_rate_limited(
    mock_update, mock_orchestrator, mock_bot, daily_quota
):
    rate_limiter = RateLimiter(max_messages=1, window_seconds=60)
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": rate_limiter,
        "daily_quota": daily_quota,
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


def test_strip_markdown_removes_headings():
    # The models keep emitting these despite the prompt, and Telegram shows
    # them as literal "###" because replies are sent without a parse_mode.
    assert _strip_markdown("### Детали\ntext") == "Детали\ntext"


def test_strip_markdown_removes_bold_but_keeps_the_words():
    assert _strip_markdown("**Издавач**: Infostan") == "Издавач: Infostan"


def test_strip_markdown_leaves_a_lone_asterisk_alone():
    # A single marker is likelier to be quoted from the document than an
    # attempt at formatting, so removing it would corrupt the answer.
    assert _strip_markdown("Формула 19=16*17*18") == "Формула 19=16*17*18"


def test_strip_markdown_keeps_plain_text_untouched():
    assert _strip_markdown("Сумма 2.066,98 РСД") == "Сумма 2.066,98 РСД"


async def test_handle_photo_strips_markdown_from_the_answer(
    mock_photo_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "document", "agent_response": "### Итог\n**Всего**: 5"}
    )

    await handle_photo(mock_photo_update, mock_context)

    reply_text = mock_photo_update.message.reply_text.call_args[0][0]
    assert reply_text == "Итог\nВсего: 5"


async def test_handle_photo_sends_the_agents_reading(mock_photo_update, mock_context):
    await handle_photo(mock_photo_update, mock_context)

    reply_text = mock_photo_update.message.reply_text.call_args[0][0]
    assert reply_text == "Test answer from orchestrator."


async def test_handle_photo_sends_the_downloaded_image(
    mock_photo_update, mock_context, mock_orchestrator
):
    # The bytes must arrive base64-encoded in a data URL, not as raw bytes,
    # and they ride in the run's context rather than in the graph's state,
    # which is what keeps them out of the checkpoint file.
    await handle_photo(mock_photo_update, mock_context)

    photo = mock_orchestrator.ainvoke.call_args.kwargs["context"]
    assert photo.image_url == "data:image/jpeg;base64,anBlZ2RhdGE="


async def test_handle_photo_passes_the_caption_to_the_graph(
    mock_photo_update, mock_context, mock_orchestrator
):
    mock_photo_update.message.caption = "Kada moram da platim?"

    await handle_photo(mock_photo_update, mock_context)

    photo = mock_orchestrator.ainvoke.call_args.kwargs["context"]
    assert photo.caption == "Kada moram da platim?"


async def test_handle_photo_sends_an_empty_caption_when_there_is_none(
    mock_photo_update, mock_context, mock_orchestrator
):
    # Telegram reports a missing caption as None, and the node decides
    # what to ask in that case, so it is handed a string either way.
    await handle_photo(mock_photo_update, mock_context)

    photo = mock_orchestrator.ainvoke.call_args.kwargs["context"]
    assert photo.caption == ""


async def test_handle_photo_takes_the_largest_photo_size(
    mock_photo_update, mock_context
):
    # Telegram attaches several sizes; small print only survives in the last.
    smallest = MagicMock()
    largest = mock_photo_update.message.photo[0]
    mock_photo_update.message.photo = [smallest, largest]

    await handle_photo(mock_photo_update, mock_context)

    largest.get_file.assert_awaited_once()


async def test_handle_photo_rejects_an_unsupported_image_type(
    mock_photo_update, mock_context, mock_orchestrator
):
    # An image sent as a file carries its own mime type, which may be one no
    # vision model reads.
    document = MagicMock()
    document.file_size = 120_000
    document.mime_type = "image/heic"
    mock_photo_update.message.photo = []
    mock_photo_update.message.document = document

    await handle_photo(mock_photo_update, mock_context)

    mock_orchestrator.ainvoke.assert_not_called()


async def test_handle_photo_does_not_download_an_oversized_image(
    mock_photo_update, mock_context
):
    # Size comes from the update itself, so the file is refused unfetched.
    photo = mock_photo_update.message.photo[0]
    photo.file_size = MAX_IMAGE_BYTES + 1

    await handle_photo(mock_photo_update, mock_context)

    photo.get_file.assert_not_called()


async def test_handle_photo_is_rate_limited(
    mock_photo_update, mock_orchestrator, mock_bot, daily_quota
):
    # A vision call costs more than a text one, so the same limit applies.
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(max_messages=1, window_seconds=60),
        "daily_quota": daily_quota,
    }

    await handle_photo(mock_photo_update, context)
    await handle_photo(mock_photo_update, context)

    assert mock_orchestrator.ainvoke.call_count == 1


async def test_handle_photo_replies_timeout_when_the_vision_call_times_out(
    mock_photo_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APITimeoutError(request=None))

    await handle_photo(mock_photo_update, mock_context)

    mock_photo_update.message.reply_text.assert_awaited_once_with(ERROR_TIMEOUT)


async def test_handle_photo_replies_generically_on_a_download_failure(
    mock_photo_update, mock_context
):
    # A Telegram failure is not an OpenAI one, so it takes the generic path.
    mock_photo_update.message.photo[0].get_file = AsyncMock(
        side_effect=TelegramError("file is gone")
    )

    await handle_photo(mock_photo_update, mock_context)

    mock_photo_update.message.reply_text.assert_awaited_once_with(ERROR_GENERIC)


async def test_unsupported_file_names_the_type_it_cannot_read(
    mock_update, mock_context
):
    # Naming the type is the point: "I can't read DOCX files" explains the
    # refusal, where silence just looks like a broken bot.
    mock_update.message.document = MagicMock()
    mock_update.message.document.file_name = "ugovor.docx"

    await handle_unsupported_file(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == UNSUPPORTED_FILE.format(kind="DOCX")


async def test_unsupported_file_copes_with_a_nameless_upload(mock_update, mock_context):
    mock_update.message.document = MagicMock()
    mock_update.message.document.file_name = None

    await handle_unsupported_file(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == UNSUPPORTED_FILE.format(kind="that kind of")


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


async def test_error_handler_classifies_an_openai_timeout(mock_update):
    # It must run the same failure mapping as a handler's own try/except,
    # not fall back to the generic apology for every error that reaches it.
    context = MagicMock()
    context.error = APITimeoutError(request=None)
    mock_update.effective_message = mock_update.message

    await error_handler(mock_update, context)

    mock_update.message.reply_text.assert_awaited_once_with(ERROR_TIMEOUT)


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


def _large_document(mock_photo_update, size=3 * 1024 * 1024):
    """Turn the photo fixture into a large uncompressed file upload."""
    document = MagicMock()
    document.file_size = size
    document.mime_type = "image/jpeg"
    document.get_file = mock_photo_update.message.photo[0].get_file
    mock_photo_update.message.photo = []
    mock_photo_update.message.document = document
    return document


async def test_handle_photo_asks_before_reading_a_large_file(
    mock_photo_update, mock_context, mock_orchestrator
):
    # Only an uncompressed upload can be big enough to matter: Telegram
    # shrinks anything sent as a photo, so that path is never asked about.
    _large_document(mock_photo_update)

    await handle_photo(mock_photo_update, mock_context)

    mock_orchestrator.ainvoke.assert_not_called()


async def test_the_large_file_question_offers_both_choices(
    mock_photo_update, mock_context
):
    _large_document(mock_photo_update)

    await handle_photo(mock_photo_update, mock_context)

    markup = mock_photo_update.message.reply_text.call_args.kwargs["reply_markup"]
    assert [button.callback_data for button in markup.inline_keyboard[0]] == [
        "doc:read",
        "doc:cancel",
    ]


async def test_the_large_file_question_replies_to_the_upload(
    mock_photo_update, mock_context
):
    # The reply link is load-bearing, exactly as for the feedback buttons:
    # it is how the tap finds the file again, with nothing stored meanwhile.
    _large_document(mock_photo_update)

    await handle_photo(mock_photo_update, mock_context)

    kwargs = mock_photo_update.message.reply_text.call_args.kwargs
    assert kwargs["reply_to_message_id"] == mock_photo_update.message.message_id


async def test_an_ordinary_photo_is_read_without_asking(
    mock_photo_update, mock_context, mock_orchestrator
):
    # A compressed photo is cheap, so a question about it would be noise.
    await handle_photo(mock_photo_update, mock_context)

    mock_orchestrator.ainvoke.assert_called_once()


async def test_confirming_reads_the_file_that_was_asked_about(
    mock_photo_update, mock_context, mock_orchestrator
):
    upload = mock_photo_update.message
    _large_document(mock_photo_update)
    query = MagicMock()
    query.data = "doc:read"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.is_accessible = True
    query.message.reply_to_message = upload
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    mock_orchestrator.ainvoke.assert_called_once()


async def test_cancelling_does_not_read_the_file(
    mock_photo_update, mock_context, mock_orchestrator
):
    query = MagicMock()
    query.data = "doc:cancel"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    mock_orchestrator.ainvoke.assert_not_called()


async def test_cancelling_says_what_to_do_instead(mock_context):
    query = MagicMock()
    query.data = "doc:cancel"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.reply_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    query.answer.assert_awaited_once_with(LARGE_FILE_CANCELLED)


async def test_confirming_an_unreachable_file_says_so(mock_context, mock_orchestrator):
    # Past ~48 hours Telegram sends an InaccessibleMessage, so the reply
    # chain back to the upload is gone and there is nothing to read.
    query = MagicMock()
    query.data = "doc:read"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.is_accessible = False
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    mock_orchestrator.ainvoke.assert_not_called()


async def test_the_large_file_question_states_the_size(mock_photo_update, mock_context):
    _large_document(mock_photo_update, size=3 * 1024 * 1024)

    await handle_photo(mock_photo_update, mock_context)

    assert "3.0 MB" in mock_photo_update.message.reply_text.call_args[0][0]


async def test_a_second_tap_does_not_read_the_file_again(
    mock_photo_update, mock_context, mock_orchestrator
):
    # Telegram refuses to clear a keyboard that is already gone, and that
    # refusal is the only signal that another tap got here first. Without it
    # a quick double tap pays for the same file twice.
    upload = mock_photo_update.message
    _large_document(mock_photo_update)
    query = MagicMock()
    query.data = "doc:read"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock(side_effect=BadRequest("not modified"))
    query.message.is_accessible = True
    query.message.reply_to_message = upload
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    mock_orchestrator.ainvoke.assert_not_called()


@patch("src.bot.handlers.clear_history", return_value=4)
async def test_reset_command_forgets_the_conversation(
    mock_clear, mock_update, mock_context
):
    await reset_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == RESET_DONE


@patch("src.bot.handlers.clear_history", return_value=0)
async def test_reset_command_says_when_there_was_nothing_to_forget(
    mock_clear, mock_update, mock_context
):
    await reset_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == RESET_EMPTY


@patch("src.bot.handlers.clear_history", return_value=4)
async def test_reset_command_points_at_pref_clear_for_preferences(
    mock_clear, mock_update, mock_context
):
    # Wiping history silently while leaving rules in place would be
    # confusing, so the reply says which one it did.
    await reset_command(mock_update, mock_context)

    assert "/pref clear" in mock_update.message.reply_text.call_args[0][0]


@patch("src.bot.handlers.get_history", return_value=[])
async def test_export_says_when_there_is_nothing_to_send(
    mock_history, mock_update, mock_context
):
    mock_context.args = []

    await export_command(mock_update, mock_context)

    assert mock_update.message.reply_text.call_args[0][0] == EXPORT_EMPTY


@patch("src.bot.handlers.get_history")
async def test_export_sends_a_file_not_a_message(
    mock_history, mock_update, mock_context
):
    # A few document readings run well past Telegram's 4096-character limit
    # for a message, so a text reply would be rejected outright.
    mock_history.return_value = [HumanMessage(content="hi"), AIMessage(content="yo")]
    mock_context.args = []
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    mock_update.message.reply_document.assert_awaited_once()


@patch("src.bot.handlers.get_history")
async def test_export_names_the_file_after_the_chat(
    mock_history, mock_update, mock_context
):
    mock_history.return_value = [HumanMessage(content="hi")]
    mock_context.args = []
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    filename = mock_update.message.reply_document.call_args.kwargs["filename"]
    assert filename == f"settlein-chat-{mock_update.message.chat_id}.txt"


@patch("src.bot.handlers.get_history")
async def test_export_writes_the_transcript_into_the_file(
    mock_history, mock_update, mock_context
):
    mock_history.return_value = [HumanMessage(content="hi"), AIMessage(content="yo")]
    mock_context.args = []
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    sent = mock_update.message.reply_document.call_args.kwargs["document"]
    assert sent.getvalue().decode("utf-8") == "[1] user: hi\n[2] bot : yo"


@patch("src.bot.handlers.get_history")
async def test_export_honours_a_requested_count(
    mock_history, mock_update, mock_context
):
    mock_history.return_value = [HumanMessage(content=str(n)) for n in range(10)]
    mock_context.args = ["3"]
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    sent = mock_update.message.reply_document.call_args.kwargs["document"]
    assert len(sent.getvalue().decode("utf-8").splitlines()) == 3


async def test_export_command_replies_unavailable_when_switched_off(
    mock_update, mock_context
):
    mock_context.bot_data["features"] = {"export": False}

    await export_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == FEATURE_UNAVAILABLE.format(name=FEATURE_EXPORT)


@patch("src.bot.handlers.get_history")
async def test_export_command_does_not_read_history_when_switched_off(
    mock_history, mock_update, mock_context
):
    mock_context.bot_data["features"] = {"export": False}

    await export_command(mock_update, mock_context)

    mock_history.assert_not_called()


@patch("src.bot.handlers.get_history")
async def test_export_command_still_works_when_its_switch_is_on(
    mock_history, mock_update, mock_context
):
    mock_history.return_value = [HumanMessage(content="hi")]
    mock_context.args = []
    mock_context.bot_data["features"] = {"export": True}
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    mock_update.message.reply_document.assert_awaited_once()


@patch("src.bot.handlers.get_history")
async def test_export_command_works_when_no_features_key_is_stored(
    mock_history, mock_update, mock_context
):
    mock_history.return_value = [HumanMessage(content="hi")]
    mock_context.args = []
    mock_update.message.reply_document = AsyncMock()

    await export_command(mock_update, mock_context)

    mock_update.message.reply_document.assert_awaited_once()


async def test_reset_command_replies_unavailable_when_switched_off(
    mock_update, mock_context
):
    mock_context.bot_data["features"] = {"reset": False}

    await reset_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == FEATURE_UNAVAILABLE.format(name=FEATURE_RESET)


@patch("src.bot.handlers.clear_history")
async def test_reset_command_does_not_clear_history_when_switched_off(
    mock_clear, mock_update, mock_context
):
    mock_context.bot_data["features"] = {"reset": False}

    await reset_command(mock_update, mock_context)

    mock_clear.assert_not_called()


@patch("src.bot.handlers.clear_history", return_value=4)
async def test_reset_command_still_works_when_its_switch_is_on(
    mock_clear, mock_update, mock_context
):
    mock_context.bot_data["features"] = {"reset": True}

    await reset_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == RESET_DONE


@patch("src.bot.handlers.clear_history", return_value=4)
async def test_reset_command_works_when_no_features_key_is_stored(
    mock_clear, mock_update, mock_context
):
    await reset_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == RESET_DONE


def test_export_limit_ignores_a_non_number():
    # "/export please" should behave like a bare "/export", not crash.
    assert _export_limit(["please"]) == EXPORT_LIMIT


def test_a_short_answer_is_sent_whole():
    assert _split_for_telegram("short answer") == ["short answer"]


def test_a_long_answer_is_split_within_the_limit():
    # Telegram rejects an over-long message outright, so a long document
    # reading arrived as "Message is too long" and no answer at all.
    long_answer = "\n".join("line %d" % n for n in range(2000))

    parts = _split_for_telegram(long_answer)

    assert all(len(part) <= TELEGRAM_MESSAGE_LIMIT for part in parts)


def test_splitting_prefers_a_line_break():
    # Cutting mid-row would split a table or a numbered list in half.
    long_answer = "\n".join("line %d" % n for n in range(2000))

    parts = _split_for_telegram(long_answer)

    assert all(part.startswith("line ") for part in parts)


def test_splitting_loses_nothing():
    long_answer = "\n".join("line %d" % n for n in range(2000))

    rejoined = "\n".join(_split_for_telegram(long_answer))

    assert rejoined == long_answer


def test_splitting_copes_with_one_unbroken_line():
    # No newline to cut at, so the hard limit applies rather than looping.
    parts = _split_for_telegram("x" * 10000)

    assert [len(part) for part in parts] == [4096, 4096, 1808]


async def test_a_long_answer_reaches_the_user_in_pieces(
    mock_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "knowledge_question", "agent_response": "x" * 9000}
    )
    mock_update.message.text = "tell me everything"

    await handle_message(mock_update, mock_context)

    assert mock_update.message.reply_text.call_count == 3


async def test_only_the_last_piece_carries_the_feedback_buttons(
    mock_update, mock_context, mock_orchestrator
):
    # The buttons mark where the answer ends, and a later tap walks back from
    # that message to find the question.
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "knowledge_question", "agent_response": "x" * 9000}
    )
    mock_update.message.text = "tell me everything"

    await handle_message(mock_update, mock_context)

    markups = [
        call.kwargs["reply_markup"]
        for call in mock_update.message.reply_text.call_args_list
    ]
    assert [markup is None for markup in markups] == [True, True, False]


async def test_a_long_photo_reading_is_split_too(
    mock_photo_update, mock_context, mock_orchestrator
):
    mock_orchestrator.ainvoke = AsyncMock(
        return_value={"intent": "document", "agent_response": "y" * 9000}
    )

    await handle_photo(mock_photo_update, mock_context)

    assert mock_photo_update.message.reply_text.call_count == 3


async def test_help_lists_the_reset_and_export_commands(mock_update, mock_context):
    # A command nobody is told about may as well not exist.
    await help_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert "/reset" in reply and "/export" in reply


async def test_handle_message_shows_the_typing_indicator(mock_update, mock_context):
    mock_update.message.text = "How do I get a work permit?"

    await handle_message(mock_update, mock_context)

    mock_context.bot.send_chat_action.assert_awaited_once_with(
        chat_id=mock_update.message.chat_id, action=ChatAction.TYPING
    )


async def test_handle_photo_shows_the_typing_indicator(mock_photo_update, mock_context):
    await handle_photo(mock_photo_update, mock_context)

    mock_context.bot.send_chat_action.assert_awaited_once_with(
        chat_id=mock_photo_update.message.chat_id, action=ChatAction.TYPING
    )


async def test_a_rate_limited_message_shows_no_typing_indicator(
    mock_update, mock_orchestrator, mock_bot
):
    # The indicator sits after the rate-limit check, so a future edit that
    # moves it earlier would start billing typing actions for a rejection.
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(max_messages=0, window_seconds=60),
    }
    mock_update.message.text = "hello"

    await handle_message(mock_update, context)

    mock_bot.send_chat_action.assert_not_awaited()


@patch("src.bot.handlers.clear_history", return_value=4)
async def test_reset_command_shows_no_typing_indicator(
    mock_clear, mock_update, mock_context
):
    # /reset answers straight from the checkpointer, with no model call.
    await reset_command(mock_update, mock_context)

    mock_context.bot.send_chat_action.assert_not_awaited()


@patch("src.bot.handlers.get_history", return_value=[])
async def test_export_command_shows_no_typing_indicator(
    mock_history, mock_update, mock_context
):
    # /export answers straight from the checkpointer, with no model call.
    mock_context.args = []

    await export_command(mock_update, mock_context)

    mock_context.bot.send_chat_action.assert_not_awaited()


@patch("src.bot.handlers.get_preferences", return_value=[])
async def test_pref_command_with_no_args_shows_no_typing_indicator(
    mock_get, mock_update, mock_context
):
    # A bare /pref only lists saved rules, with no model call.
    mock_context.args = []

    await pref_command(mock_update, mock_context)

    mock_context.bot.send_chat_action.assert_not_awaited()


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_loglevel_command_is_silent_outside_the_admin_chat(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.args = ["warning"]
    mock_update.message.reply_document = AsyncMock()

    await loglevel_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()
    mock_update.message.reply_document.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_logs_command_is_silent_outside_the_admin_chat(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_update.message.reply_document = AsyncMock()

    await logs_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()
    mock_update.message.reply_document.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_command_is_silent_when_there_is_no_log_handler(
    mock_is_admin, mock_update, mock_context
):
    # Only reachable if the mirror failed to install - the mock leaves
    # bot_data without a "log_handler" key entirely.
    mock_context.args = ["warning"]

    await loglevel_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_logs_command_is_silent_when_there_is_no_log_handler(
    mock_is_admin, mock_update, mock_context
):
    await logs_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_warning_moves_the_push_level(
    mock_is_admin, mock_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_context.args = ["warning"]

    await loglevel_command(mock_update, mock_context)

    assert handler.push_level == logging.WARNING


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_warning_confirms_the_change(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.args = ["warning"]

    await loglevel_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == LOGLEVEL_SET.format(level="WARNING")


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_with_an_unknown_level_shows_usage(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.args = ["nonsense"]

    await loglevel_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == LOGLEVEL_USAGE.format(current="ERROR")


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_with_an_unknown_level_leaves_the_push_level_unchanged(
    mock_is_admin, mock_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_context.args = ["nonsense"]

    await loglevel_command(mock_update, mock_context)

    assert handler.push_level == logging.ERROR


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_with_no_args_shows_usage(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.args = []

    await loglevel_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == LOGLEVEL_USAGE.format(current="ERROR")


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_loglevel_with_no_args_leaves_the_push_level_unchanged(
    mock_is_admin, mock_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_context.args = []

    await loglevel_command(mock_update, mock_context)

    assert handler.push_level == logging.ERROR


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_logs_command_replies_when_the_buffer_is_empty(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()

    await logs_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(LOGS_EMPTY)


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_logs_command_sends_a_document_when_there_are_records(
    mock_is_admin, mock_update, mock_context
):
    handler = TelegramLogHandler()
    handler.handle(
        logging.LogRecord(
            name="src.bot.handlers",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="boom",
            args=(),
            exc_info=None,
        )
    )
    mock_context.bot_data["log_handler"] = handler
    mock_update.message.reply_document = AsyncMock()

    await logs_command(mock_update, mock_context)

    mock_update.message.reply_document.assert_awaited_once()


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_admin_command_is_silent_outside_the_admin_chat(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()

    await admin_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_is_silent_when_there_is_no_log_handler(
    mock_is_admin, mock_update, mock_context
):
    await admin_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_panel_marks_the_level_currently_in_force(
    mock_is_admin, mock_update, mock_context
):
    # push_level is moved away from the handler's own default so the
    # assertion cannot pass merely by reading a fresh handler's starting
    # value - it proves the panel reads live state, not a copy.
    handler = TelegramLogHandler()
    handler.push_level = logging.WARNING
    mock_context.bot_data["log_handler"] = handler

    await admin_command(mock_update, mock_context)

    keyboard = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    labels = [button.text for button in keyboard.inline_keyboard[0]]
    assert labels == ["● WARNING", "○ ERROR"]


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_moves_the_push_level_to_the_tapped_level(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:warning"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.WARNING


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_redraws_the_panel_in_place(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:warning"

    await admin_callback(mock_admin_callback_update, mock_context)

    state = _PanelState(
        push_level=logging.WARNING,
        features=default_features(),
        limits_report=_limits_report(mock_context.bot_data["daily_quota"]),
    )
    redraw = mock_admin_callback_update.callback_query.edit_message_text
    redraw.assert_awaited_once_with(
        _admin_panel_text(state), reply_markup=_admin_keyboard(state)
    )


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_admin_callback_from_outside_the_admin_chat_changes_nothing(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:warning"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.ERROR


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_ignores_an_unknown_level_payload(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:nonsense"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.ERROR


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_ignores_a_payload_that_is_not_its_own(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # Shaped like a button payload, but issued by the feedback keyboard.
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "fb:up:knowledge_question"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.ERROR


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_still_answers_an_unrecognised_payload(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:nonsense"

    await admin_callback(mock_admin_callback_update, mock_context)

    mock_admin_callback_update.callback_query.answer.assert_awaited_once_with()


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_still_moves_the_level_when_the_redraw_is_refused(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # Telegram refuses an edit that would leave the message unchanged, or
    # whose message has been deleted - either way the tap already landed.
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:warning"
    mock_admin_callback_update.callback_query.edit_message_text = AsyncMock(
        side_effect=BadRequest("message is not modified")
    )

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.WARNING


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_still_moves_the_level_when_the_panel_is_too_old(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # python-telegram-bot raises TypeError, not BadRequest, for a panel past
    # about 48 hours - an InaccessibleMessage carries none of the attributes
    # edit_message_text needs.
    handler = TelegramLogHandler()
    mock_context.bot_data["log_handler"] = handler
    mock_admin_callback_update.callback_query.data = "adm:warning"
    mock_admin_callback_update.callback_query.edit_message_text = AsyncMock(
        side_effect=TypeError("message is inaccessible")
    )

    await admin_callback(mock_admin_callback_update, mock_context)

    assert handler.push_level == logging.WARNING


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_names_the_new_level_when_the_redraw_is_refused(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:warning"
    mock_admin_callback_update.callback_query.edit_message_text = AsyncMock(
        side_effect=BadRequest("message is not modified")
    )

    await admin_callback(mock_admin_callback_update, mock_context)

    mock_admin_callback_update.callback_query.answer.assert_awaited_once_with(
        LOGLEVEL_SET.format(level="WARNING")
    )


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_names_the_new_level_when_the_panel_is_too_old(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:warning"
    mock_admin_callback_update.callback_query.edit_message_text = AsyncMock(
        side_effect=TypeError("message is inaccessible")
    )

    await admin_callback(mock_admin_callback_update, mock_context)

    mock_admin_callback_update.callback_query.answer.assert_awaited_once_with(
        LOGLEVEL_SET.format(level="WARNING")
    )


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_panel_shows_the_position_of_each_switch(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.bot_data["features"] = {"export": True, "reset": False}

    await admin_command(mock_update, mock_context)

    keyboard = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    labels = [button.text for button in keyboard.inline_keyboard[1]]
    assert labels == ["/export: on", "/reset: off"]


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_switch_button_names_the_position_a_tap_sets(
    mock_is_admin, mock_update, mock_context
):
    # The payload names the target position, not a flip, so a tap on a
    # panel someone else has already moved still lands where it says.
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_context.bot_data["features"] = {"export": True}

    await admin_command(mock_update, mock_context)

    keyboard = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    button = keyboard.inline_keyboard[1][0]
    assert button.callback_data == "adm:export:off"


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_writes_the_tapped_position_into_bot_data(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:export:off"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert mock_context.bot_data["features"]["export"] is False


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_applying_the_same_switch_tap_twice_is_stable(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # A stale panel tapped twice (once by each of two operators) must not
    # flip back and forth - the payload names a position, not a toggle.
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:export:off"

    await admin_callback(mock_admin_callback_update, mock_context)
    await admin_callback(mock_admin_callback_update, mock_context)

    assert mock_context.bot_data["features"]["export"] is False


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_rejects_a_payload_missing_a_position(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # Shaped like the step-1 panel's payload, before switches existed.
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    mock_admin_callback_update.callback_query.data = "adm:export"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert "features" not in mock_context.bot_data


@patch(
    "src.bot.handlers.tidy_preferences",
    new=AsyncMock(return_value=["Merged rule."]),
)
async def test_pref_tidy_shows_the_typing_indicator(mock_update, mock_context):
    # The one /pref branch that waits on a model, and the only indicator site
    # that lives inside a helper rather than in a handler of its own - the
    # three tests above it would all still pass if it were dropped.
    mock_context.args = ["tidy"]
    mock_context.bot_data["preference_tidier"] = MagicMock()

    await pref_command(mock_update, mock_context)

    mock_context.bot.send_chat_action.assert_awaited_once_with(
        chat_id=mock_update.message.chat_id, action=ChatAction.TYPING
    )


def _context_with_quota(mock_orchestrator, mock_bot, quota):
    """Build a bot_data dict around a DailyQuota a test wants to exhaust."""
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(),
        "daily_quota": quota,
    }
    return context


async def test_handle_message_refuses_a_chat_past_the_text_allowance(
    mock_update, mock_orchestrator, mock_bot
):
    context = _context_with_quota(
        mock_orchestrator, mock_bot, DailyQuota(text_limit=0, photo_limit=5)
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == QUOTA_EXHAUSTED[KIND_TEXT].format(limit=0)
    mock_orchestrator.ainvoke.assert_not_called()


async def test_handle_message_records_one_message_on_a_delivered_answer(
    mock_update, mock_context
):
    quota = mock_context.bot_data["daily_quota"]
    mock_update.message.text = "What is a White Card?"

    await handle_message(mock_update, mock_context)

    assert quota.usage(mock_update.message.chat_id)[KIND_TEXT] == 1


async def test_handle_message_records_nothing_when_the_orchestrator_raises(
    mock_update, mock_context, mock_orchestrator
):
    # The allowance pays for answers, not attempts: a failed call must not
    # burn the chat's daily budget.
    quota = mock_context.bot_data["daily_quota"]
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APITimeoutError(request=None))
    mock_update.message.text = "test question"

    await handle_message(mock_update, mock_context)

    assert quota.usage(mock_update.message.chat_id)[KIND_TEXT] == 0


async def test_handle_photo_refuses_a_chat_past_the_photo_allowance(
    mock_photo_update, mock_orchestrator, mock_bot
):
    context = _context_with_quota(
        mock_orchestrator, mock_bot, DailyQuota(text_limit=5, photo_limit=0)
    )

    await handle_photo(mock_photo_update, context)

    reply = mock_photo_update.message.reply_text.call_args[0][0]
    assert reply == QUOTA_EXHAUSTED[KIND_PHOTO].format(limit=0)
    mock_orchestrator.ainvoke.assert_not_called()


async def test_handle_photo_records_nothing_when_the_vision_call_fails(
    mock_photo_update, mock_context, mock_orchestrator
):
    quota = mock_context.bot_data["daily_quota"]
    mock_orchestrator.ainvoke = AsyncMock(side_effect=APITimeoutError(request=None))

    await handle_photo(mock_photo_update, mock_context)

    assert quota.usage(mock_photo_update.message.chat_id)[KIND_PHOTO] == 0


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_handle_message_serves_the_admin_chat_past_the_text_allowance(
    mock_is_admin, mock_update, mock_orchestrator, mock_bot
):
    context = _context_with_quota(
        mock_orchestrator, mock_bot, DailyQuota(text_limit=0, photo_limit=0)
    )
    mock_update.message.text = "test question"

    await handle_message(mock_update, context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == "Test answer from orchestrator."


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_handle_message_records_nothing_for_the_exempt_admin_chat(
    mock_is_admin, mock_update, mock_orchestrator, mock_bot
):
    quota = DailyQuota(text_limit=0, photo_limit=0)
    context = _context_with_quota(mock_orchestrator, mock_bot, quota)
    mock_update.message.text = "test question"

    await handle_message(mock_update, context)

    assert quota.usage(mock_update.message.chat_id)[KIND_TEXT] == 0


async def test_handle_photo_refuses_a_large_file_before_offering_to_read_it(
    mock_photo_update, mock_orchestrator, mock_bot
):
    # Tapping "Read it" clears the keyboard, so a refusal discovered only
    # after the tap would take the offer away with it. The allowance has to
    # be checked, and can refuse, before the buttons are ever sent.
    _large_document(mock_photo_update)
    context = _context_with_quota(
        mock_orchestrator, mock_bot, DailyQuota(text_limit=5, photo_limit=0)
    )

    await handle_photo(mock_photo_update, context)

    assert mock_photo_update.message.reply_text.call_args.kwargs == {}


async def test_confirming_a_large_file_records_it_exactly_once(
    mock_photo_update, mock_context, mock_orchestrator
):
    quota = mock_context.bot_data["daily_quota"]
    upload = mock_photo_update.message
    _large_document(mock_photo_update)
    query = MagicMock()
    query.data = "doc:read"
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    query.message.is_accessible = True
    query.message.reply_to_message = upload
    update = MagicMock()
    update.callback_query = query

    await document_callback(update, mock_context)

    assert quota.usage(upload.chat_id)[KIND_PHOTO] == 1


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_limits_command_shows_a_user_their_own_allowance(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = []
    quota = DailyQuota(text_limit=30, photo_limit=5)
    quota.record(mock_update.message.chat_id, KIND_TEXT)
    mock_context.bot_data["daily_quota"] = quota

    await limits_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_awaited_once_with(
        LIMITS_PERSONAL.format(text=30, photo=5, text_used=1, photo_used=0)
    )


@patch("src.bot.handlers.is_admin", return_value=False)
async def test_limits_command_will_not_move_a_limit_for_an_ordinary_user(
    mock_is_admin, mock_update, mock_context
):
    before = mock_context.bot_data["daily_quota"].limits[KIND_TEXT]
    mock_context.args = ["text", "999"]

    await limits_command(mock_update, mock_context)

    assert mock_context.bot_data["daily_quota"].limits[KIND_TEXT] == before


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_with_no_args_shows_the_status(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = []

    await limits_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == _limits_status(mock_context.bot_data["daily_quota"])


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_text_moves_the_text_allowance(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = ["text", "30"]

    await limits_command(mock_update, mock_context)

    assert mock_context.bot_data["daily_quota"].limits[KIND_TEXT] == 30


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_photo_moves_the_photo_allowance(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = ["photo", "5"]

    await limits_command(mock_update, mock_context)

    assert mock_context.bot_data["daily_quota"].limits[KIND_PHOTO] == 5


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_with_a_bad_number_replies_limits_bad_args(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = ["text", "abc"]

    await limits_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == LIMITS_BAD_ARGS


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_with_a_bad_number_changes_nothing(
    mock_is_admin, mock_update, mock_context
):
    quota = mock_context.bot_data["daily_quota"]
    original = quota.limits[KIND_TEXT]
    mock_context.args = ["text", "abc"]

    await limits_command(mock_update, mock_context)

    assert quota.limits[KIND_TEXT] == original


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_replies_with_the_new_status_after_a_change(
    mock_is_admin, mock_update, mock_context
):
    mock_context.args = ["text", "30"]

    await limits_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == _limits_status(mock_context.bot_data["daily_quota"])


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_limits_command_with_no_change_replies_with_the_status_only(
    mock_is_admin, mock_update, mock_context
):
    quota = mock_context.bot_data["daily_quota"]
    mock_context.args = ["text", str(quota.limits[KIND_TEXT])]

    await limits_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert reply == _limits_status(quota)


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_panel_text_states_the_daily_allowances(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    quota = mock_context.bot_data["daily_quota"]

    await admin_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert f"{quota.limits[KIND_TEXT]} questions" in reply


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_panel_text_states_todays_spend(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    quota = mock_context.bot_data["daily_quota"]
    quota.record(chat_id=999, kind=KIND_TEXT)

    await admin_command(mock_update, mock_context)

    reply = mock_update.message.reply_text.call_args[0][0]
    assert "Spent today: 1 questions and 0 photos across 1 chats." in reply


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_command_keyboard_has_the_reset_counters_button(
    mock_is_admin, mock_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()

    await admin_command(mock_update, mock_context)

    keyboard = mock_update.message.reply_text.call_args.kwargs["reply_markup"]
    labels = [button.text for row in keyboard.inline_keyboard for button in row]
    assert RESET_COUNTERS_LABEL in labels


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_counters_tap_zeroes_the_quota(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    quota = mock_context.bot_data["daily_quota"]
    quota.record(chat_id=1, kind=KIND_TEXT)
    mock_admin_callback_update.callback_query.data = "adm:counters"

    await admin_callback(mock_admin_callback_update, mock_context)

    assert quota.spent_today() == {KIND_TEXT: 0, KIND_PHOTO: 0}


@patch("src.bot.handlers.is_admin", return_value=True)
async def test_admin_callback_counters_tap_redraws_the_panel_with_the_zeroed_spend(
    mock_is_admin, mock_admin_callback_update, mock_context
):
    # The spend line is what makes the redraw genuinely change the message:
    # without it, resetting the counters would leave the panel text
    # identical and Telegram would refuse the edit as "not modified".
    mock_context.bot_data["log_handler"] = TelegramLogHandler()
    quota = mock_context.bot_data["daily_quota"]
    quota.record(chat_id=1, kind=KIND_TEXT)
    mock_admin_callback_update.callback_query.data = "adm:counters"

    await admin_callback(mock_admin_callback_update, mock_context)

    redraw = mock_admin_callback_update.callback_query.edit_message_text
    redrawn_text = redraw.call_args.args[0]
    assert "Spent today: 0 questions and 0 photos across 0 chats." in redrawn_text
