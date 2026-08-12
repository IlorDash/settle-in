from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import CallbackQuery, Chat, Message, Update, User

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
def mock_orchestrator():
    orchestrator = MagicMock()
    orchestrator.ainvoke = AsyncMock(
        return_value={
            "user_message": "test",
            "intent": "knowledge_question",
            "agent_response": "Test answer from orchestrator.",
        }
    )
    return orchestrator


@pytest.fixture
def mock_context(mock_orchestrator):
    context = MagicMock()
    context.bot_data = {
        "orchestrator": mock_orchestrator,
        "rate_limiter": RateLimiter(),
    }
    return context
