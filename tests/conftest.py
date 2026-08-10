from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Chat, Message, Update, User

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
