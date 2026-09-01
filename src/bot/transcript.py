"""Render a chat's stored messages as readable text, for /export."""

from langchain_core.messages import AIMessage


def speaker(message) -> str:
    """Label one stored message by who sent it.

    Args:
        message: A stored LangChain message.

    Returns:
        A fixed-width label, so the transcript's colons line up.
    """
    return "bot " if isinstance(message, AIMessage) else "user"


def format_transcript(messages: list, limit: int) -> str:
    """Render the last `limit` messages as a readable transcript.

    Args:
        messages: The chat's stored messages, oldest first.
        limit: How many of the most recent ones to keep; 0 keeps all.

    Returns:
        One line per message, numbered from the start of the whole history so
        that two exports of the same chat can be lined up against each other.
    """
    kept = messages[-limit:] if limit else messages
    first = len(messages) - len(kept) + 1
    return "\n".join(
        f"[{number}] {speaker(message)}: {message.content}"
        for number, message in enumerate(kept, first)
    )
