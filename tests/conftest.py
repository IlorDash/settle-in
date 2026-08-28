from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import (
    Bot,
    CallbackQuery,
    Chat,
    File,
    Message,
    PhotoSize,
    Update,
    User,
)

from src.bot.middleware import RateLimiter


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = 12345
    user.first_name = "Test"
    user.is_bot = False
    return user


@pytest.fixture
def mock_chat():
    chat = MagicMock(spec=Chat)
    chat.id = 12345
    chat.type = "private"
    return chat


@pytest.fixture
def mock_update(mock_user, mock_chat):
    update = MagicMock(spec=Update)
    update.message = MagicMock(spec=Message)
    update.message.from_user = mock_user
    update.message.chat = mock_chat
    update.message.chat_id = mock_chat.id
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_callback_update(mock_user):
    """An Update carrying a feedback button tap on an answer.

    The tapped message is the bot's answer, and its reply_to_message is the
    question that produced it — the chain the handler reads instead of
    keeping state between the answer and the tap.
    """
    question = MagicMock(spec=Message)
    question.text = "What is a White Card?"

    answer = MagicMock(spec=Message)
    answer.reply_to_message = question

    query = MagicMock(spec=CallbackQuery)
    query.data = "fb:up:knowledge_question"
    query.from_user = mock_user
    query.message = answer
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()

    update = MagicMock(spec=Update)
    update.callback_query = query
    return update


@pytest.fixture
def mock_admin_callback_update(mock_chat):
    """An Update carrying a tap on the operator panel.

    `_is_admin_tap` reads the chat off `message.chat`, not the `chat_id`
    shortcut, since a stale panel arrives as an InaccessibleMessage that
    carries a chat but none of the shortcuts Message defines on top of it -
    so the mock is built the same way rather than setting `chat_id` directly.
    """
    message = MagicMock(spec=Message)
    message.chat = mock_chat

    query = MagicMock(spec=CallbackQuery)
    query.data = "adm:warning"
    query.message = message
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.callback_query = query
    return update


@pytest.fixture
def mock_photo_update(mock_user, mock_chat):
    """An Update carrying a photographed document, ready to be downloaded.

    Telegram sends a photo at several sizes and the handler takes the last;
    the file behind it yields its bytes through download_as_bytearray.
    """
    telegram_file = MagicMock(spec=File)
    telegram_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"jpegdata"))

    photo = MagicMock(spec=PhotoSize)
    photo.file_size = 120_000
    photo.get_file = AsyncMock(return_value=telegram_file)

    message = MagicMock(spec=Message)
    message.photo = [photo]
    message.document = None
    message.caption = None
    message.from_user = mock_user
    message.chat = mock_chat
    message.chat_id = mock_chat.id
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message
    return update


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.ainvoke = AsyncMock(
        return_value={
            "user_message": "test",
            "intent": "knowledge_question",
            "agent_response": "Test answer from orchestrator.",
        }
    )
    # The preference helpers read the checkpointer for real, so the snapshot
    # has to hold a plain dict — a bare MagicMock is not iterable.
    orchestrator.get_state.return_value.values = {}
    return orchestrator


@pytest.fixture
def mock_bot():
    """The Bot handlers reach through `context.bot`.

    send_chat_action has to be awaitable: every path that waits on a model
    shows the typing indicator, and a plain MagicMock cannot be awaited.
    """
    bot = MagicMock(spec=Bot)
    bot.send_chat_action = AsyncMock()
    return bot


@pytest.fixture
def mock_context(mock_orchestrator, mock_bot):
    context = MagicMock()
    context.bot = mock_bot
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(),
    }
    return context
