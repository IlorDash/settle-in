from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Chat, Message, Update, User


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
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_rag_chain():
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value="Test answer from RAG.")
    return chain


@pytest.fixture
def mock_context(mock_rag_chain):
    context = MagicMock()
    context.bot_data = {"rag_chain": mock_rag_chain}
    return context
