from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from telegram import Chat, InaccessibleMessage
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError

from src.bot.handlers import (
    ERROR_CONNECTION,
    ERROR_GENERIC,
    ERROR_RATE_LIMIT,
    ERROR_TIMEOUT,
    EXPORT_EMPTY,
    EXPORT_LIMIT,
    FEEDBACK_LOST,
    FEEDBACK_THANKS,
    LARGE_FILE_CANCELLED,
    RESET_DONE,
    RESET_EMPTY,
    TELEGRAM_MESSAGE_LIMIT,
    UNSUPPORTED_FILE,
    _export_limit,
    _split_for_telegram,
    _strip_markdown,
    document_callback,
    error_handler,
    export_command,
    feedback_callback,
    handle_message,
    handle_photo,
    handle_unsupported_file,
    help_command,
    pref_command,
    reset_command,
    start_command,
)
from src.bot.middleware import MAX_IMAGE_BYTES, RateLimiter


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


async def test_handle_message_rejects_when_rate_limited(
    mock_update, mock_orchestrator, mock_bot
):
    rate_limiter = RateLimiter(max_messages=1, window_seconds=60)
    context = MagicMock()
    context.bot = mock_bot
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
    mock_photo_update, mock_orchestrator, mock_bot
):
    # A vision call costs more than a text one, so the same limit applies.
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(max_messages=1, window_seconds=60),
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
