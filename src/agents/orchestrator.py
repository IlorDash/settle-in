import asyncio
import re
from dataclasses import dataclass
from typing import Annotated, Literal, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from src.agents.intent_classifier import classify, load_classifier
from src.agents.multimodal_agent import (
    DEFAULT_QUESTION,
    TranscriptRequest,
    build_summary_chain,
    build_transcription_chain,
    summarise,
    transcribe,
)
from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

INTENT_KNOWLEDGE_QUESTION = "knowledge_question"
INTENT_TRANSLATION = "translation"
INTENT_OUT_OF_SCOPE = "out_of_scope"
# Not a label the classifier can produce: the document node sets it, so that
# a photo turn is distinguishable in the state. Deliberately outside the
# bot's FEEDBACK_INTENTS, since a thumb on it would teach the text
# classifier nothing.
INTENT_DOCUMENT = "document"

# What the entry edge reads. Modality is known for free - Telegram states it
# in the update - so it is settled before any model is asked anything.
MODALITY_TEXT = "text"
MODALITY_DOCUMENT = "document"

# The image is never stored, so the chat log keeps a note of it instead.
DOCUMENT_HISTORY_NOTE = "[photo of a document]"

CONFIDENCE_THRESHOLD = 0.8  # Set according to accuracy research
MAX_HISTORY_MESSAGES = 6
MAX_STORED_MESSAGES = 200
MAX_STORED_PREFERENCES = 20

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about living in Serbia and with "
    "Serbian-English and Serbian-Russian translation. Please rephrase your "
    "message around one of those."
)

CLASSIFICATION_PROMPT = (
    "You are a message classifier for an immigrant assistance bot in Serbia. "
    "Classify the user's message into one of three categories:\n\n"
    f"- {INTENT_KNOWLEDGE_QUESTION}: Questions about living in Serbia, "
    "immigration procedures, documents, laws, banks, taxes, healthcare, etc.\n"
    f"- {INTENT_TRANSLATION}: Requests to translate between Serbian, English, and "
    "Russian, or a pasted foreign sentence to translate.\n"
    f"- {INTENT_OUT_OF_SCOPE}: Anything else, including a single word or short "
    "fragment with no clear request.\n\n"
    "Use the conversation so far to resolve short follow-ups: e.g. 'and in "
    "Latin?' right after a translation is still translation.\n\n"
    "Respond with ONLY the category name, nothing else."
)

PREFERENCE_TIDY_PROMPT = (
    "You are given a user's list of standing preferences - durable rules for "
    "how an assistant should respond (mainly how to translate). Rewrite the "
    "list so that rules meaning the same thing are merged into one clear rule. "
    "Keep every distinct instruction; never invent a rule the list does not "
    "imply, and never drop a unique one. Return the cleaned rules as concise "
    "imperative sentences, one per line, with no numbering or bullets."
)

# Matches a single leading list marker (a bullet or "1." / "1)") so the tidier's
# output can be parsed without mangling a rule that merely starts with a number.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def append_capped_messages(existing: list, new: list) -> list:
    """Append new messages, then keep only the most recent MAX_STORED_MESSAGES.

    `add_messages` handles the append/merge semantics (message ids, removals);
    we then bound the list so a single long-lived chat cannot grow the stored
    state without limit.
    """
    merged = add_messages(existing or [], new)
    return merged[-MAX_STORED_MESSAGES:]


def merge_preferences(existing: dict, update: dict) -> dict:
    """Reducer that merges preference updates into the stored preferences.

    A list value (the `instructions` list) replaces the stored one - the caller
    (the /pref command helpers) already merged old and new - and is
    de-duplicated and capped at MAX_STORED_PREFERENCES; scalar values are
    replaced.
    """
    merged = {**(existing or {})}
    for key, value in (update or {}).items():
        if isinstance(value, list):
            deduped = []
            for item in value:
                if item not in deduped:
                    deduped.append(item)
            merged[key] = deduped[-MAX_STORED_PREFERENCES:]
        else:
            merged[key] = value
    return merged


class OrchestratorState(TypedDict):
    """State that flows through the orchestrator graph.

    Attributes:
        messages: The running conversation log for one chat. New messages are
            appended (never overwritten) and the log is capped at the most
            recent MAX_STORED_MESSAGES so it cannot grow without bound.
        modality: Whether this turn arrived as text or as a photograph. Read
            by the entry edge to pick a branch. Every entry point sets it, so
            that the previous turn's value can never decide this one's route.
        intent: Classified intent (knowledge_question, translation, out_of_scope).
        agent_response: This turn's reply to send back to the user.
        preferences: Per-chat standing instructions the user has set via the
            /pref command (e.g. {"instructions": ["Reply in Cyrillic."]});
            persisted by the checkpointer and read by handle_translation.
    """

    messages: Annotated[list, append_capped_messages]
    modality: str
    intent: str
    agent_response: str
    preferences: Annotated[dict, merge_preferences]


@dataclass
class DocumentTurn:
    """One photographed document, as it enters the graph for a single run.

    LangGraph's runtime context rather than a field of OrchestratorState, and
    the difference is the whole reason this class exists: every state field is
    written to the checkpointer after each step, so a base64 image kept there
    would be copied into the chat's SQLite file on every photo - up to
    MAX_IMAGE_BYTES inflated by a third. Context lives only for the run and is
    never persisted. What outlives the turn is the transcript the node
    produces, which is text.

    Attributes:
        image_url: The photo as a data URL, ready for the vision model.
        caption: What the user wrote with the photo, if anything.
    """

    image_url: str = ""
    caption: str = ""


def _build_classifier_chain():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFICATION_PROMPT),
            MessagesPlaceholder("history", optional=True),
            (
                "human",
                "Classify this message: {input}\n" "Reply with only the category name.",
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def build_preference_tidier():
    """Build an LCEL chain that merges semantically-duplicate preferences.

    Used only by the on-demand /pref tidy command, never automatically, so the
    LLM's non-determinism stays out of the normal /pref add path.

    Returns:
        A Runnable taking {"current": <rules text>} that returns the cleaned
        rules, one per line.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PREFERENCE_TIDY_PROMPT),
            ("human", "Current preferences:\n{current}\n\nReturn the cleaned list."),
        ]
    )

    return prompt | llm | StrOutputParser()


def _parse_rule_lines(text: str) -> list:
    """Split an LLM's line-per-rule output into clean instruction strings."""
    rules = []
    for raw in text.splitlines():
        cleaned = _LIST_MARKER.sub("", raw).strip()
        if cleaned:
            rules.append(cleaned)
    return rules


def _recent_history(messages: list) -> list:
    """Return the last few prior turns, excluding the current message.

    Bounding the history keeps the classifier focused and the prompt small; an
    unbounded log would add noise and cost as the conversation grows.
    """
    return messages[:-1][-MAX_HISTORY_MESSAGES:]


def _preferences_directive(preferences: dict) -> str:
    """Turn the stored standing instructions into a directive for the agents.

    The wording stresses that the rules apply to every reply and every target
    language, because a language-scoped rule in the list (e.g. "Write Serbian
    in Cyrillic") otherwise biases the model into treating the whole set as
    Serbian-only.

    The closing clause keeps a one-off request ahead of a standing rule. The
    directive is a system message placed right before the user's turn, so it
    outranks and outweighs the request itself; without an explicit exception a
    stored "in Cyrillic" silently overrides someone asking for Latin now.
    """
    instructions = (preferences or {}).get("instructions") or []
    if not instructions:
        return "(no special preferences)"
    lines = "\n".join(f"- {instruction}" for instruction in instructions)
    return (
        "Apply these standing user preferences to every reply, whatever the "
        "target language. If the current message explicitly asks for something "
        "different, that request wins for this reply:\n" + lines
    )


def _document_history_note(caption: str, transcript: str) -> str:
    """Describe a photo turn in words, since the image itself is not stored.

    The transcript rides along because it is what the photo contained, so it
    belongs to the user's turn rather than the bot's answer. Later questions
    are then answered from what the page said, not from what the bot happened
    to mention about it.

    Args:
        caption: What the user wrote with the photo, or an empty string.
        transcript: The model's verbatim reading of the image.

    Returns:
        The text stored in place of the image.
    """
    asked = f"{DOCUMENT_HISTORY_NOTE} {caption}" if caption else DOCUMENT_HISTORY_NOTE
    return f"{asked}\n\n[what the photo shows]\n{transcript}"


def build_orchestrator(rag_chain, translation_chain, checkpointer=None):
    """Build and compile the LangGraph orchestrator.

    Modality decides the branch taken out of START: a photograph goes to the
    document agent, and text goes on to be classified. Intent is classified by
    the local DNN classifier, with low-confidence messages falling back to the
    LLM classifier, and routes to the RAG chain, the translation chain, or an
    out-of-scope rejection.

    Args:
        rag_chain: Compiled RAG chain from rag_agent.build_rag_chain().
        translation_chain: Compiled translation chain from
            translation_agent.build_translation_chain().
        checkpointer: Where per-chat history and preferences are stored.
            Defaults to an in-memory saver, which is what tests and the CLI
            harness want; the deployed bot passes a disk-backed one so state
            survives a restart.

    Returns:
        A compiled LangGraph that accepts OrchestratorState and returns the final state.
    """
    classifier = load_classifier()
    classifier_chain = _build_classifier_chain()
    # Built here rather than injected, for the same reason as the classifier
    # chain above: both need nothing but settings. The RAG chain is passed in
    # only because it carries a retriever over the embedded knowledge base.
    transcription_chain = build_transcription_chain()
    summary_chain = build_summary_chain()

    async def classify_intent(state: OrchestratorState) -> dict:
        message = state["messages"][-1].content
        history = _recent_history(state["messages"])
        intent, confidence = classify(classifier, message)

        # Trust the DNN when it is confident -- unless it says out_of_scope in
        # the middle of a conversation, where a lone message is more likely a
        # follow-up ("and in Latin?") that only makes sense with context.
        is_possible_followup = intent == INTENT_OUT_OF_SCOPE and bool(history)
        if confidence >= CONFIDENCE_THRESHOLD and not is_possible_followup:
            return {"intent": intent}

        # Unsure or a likely follow-up: ask the context-aware LLM classifier.
        result = await classifier_chain.ainvoke({"input": message, "history": history})
        llm_intent = result.strip().lower()
        if INTENT_TRANSLATION in llm_intent:
            return {"intent": INTENT_TRANSLATION}
        if INTENT_OUT_OF_SCOPE in llm_intent:
            return {"intent": INTENT_OUT_OF_SCOPE}
        return {"intent": INTENT_KNOWLEDGE_QUESTION}

    async def handle_knowledge_question(state: OrchestratorState) -> dict:
        message = state["messages"][-1].content
        history = _recent_history(state["messages"])
        response = await rag_chain.ainvoke({"input": message, "history": history})
        return {"agent_response": response, "messages": [AIMessage(content=response)]}

    async def handle_translation(state: OrchestratorState) -> dict:
        message = state["messages"][-1].content
        history = _recent_history(state["messages"])
        directive = _preferences_directive(state.get("preferences"))
        response = await translation_chain.ainvoke(
            {"input": message, "history": history, "preferences": directive}
        )
        return {"agent_response": response, "messages": [AIMessage(content=response)]}

    async def handle_out_of_scope(state: OrchestratorState) -> dict:
        return {
            "agent_response": OUT_OF_SCOPE_MESSAGE,
            "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)],
        }

    async def handle_document(
        state: OrchestratorState, runtime: Runtime[DocumentTurn]
    ) -> dict:
        # The photo is read once, and the reply is written from that reading
        # rather than from the image again: two readings of one page disagree
        # in places, and since only the transcript is remembered, the bot
        # would answer one way now and the other way a turn later.
        photo = runtime.context
        transcript = await transcribe(transcription_chain, photo.image_url)
        response = await summarise(
            summary_chain,
            TranscriptRequest(
                transcript=transcript,
                question=photo.caption or DEFAULT_QUESTION,
                preferences=_preferences_directive(state.get("preferences")),
            ),
        )
        return {
            "intent": INTENT_DOCUMENT,
            "agent_response": response,
            "messages": [
                HumanMessage(content=_document_history_note(photo.caption, transcript)),
                AIMessage(content=response),
            ],
        }

    def route_by_modality(
        state: OrchestratorState,
    ) -> Literal["classify_intent", "handle_document"]:
        # .get(), not [], because update_state evaluates this branch as well:
        # the /pref helpers write to the checkpointer without running the
        # graph, and a chat that has never sent a message has no modality
        # stored. Nothing marking a turn as a photograph means it is not one.
        if state.get("modality") == MODALITY_DOCUMENT:
            return "handle_document"
        return "classify_intent"

    def route_to_agent(
        state: OrchestratorState,
    ) -> Literal[
        "handle_knowledge_question", "handle_translation", "handle_out_of_scope"
    ]:
        if state["intent"] == INTENT_TRANSLATION:
            return "handle_translation"
        if state["intent"] == INTENT_OUT_OF_SCOPE:
            return "handle_out_of_scope"
        return "handle_knowledge_question"

    graph = StateGraph(OrchestratorState, context_schema=DocumentTurn)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_knowledge_question", handle_knowledge_question)
    graph.add_node("handle_translation", handle_translation)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)
    graph.add_node("handle_document", handle_document)

    graph.add_conditional_edges(START, route_by_modality)
    graph.add_conditional_edges("classify_intent", route_to_agent)
    graph.add_edge("handle_knowledge_question", END)
    graph.add_edge("handle_translation", END)
    graph.add_edge("handle_out_of_scope", END)
    graph.add_edge("handle_document", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def _thread_config(thread_id) -> dict:
    """Build the checkpointer config that scopes state to one conversation."""
    return {"configurable": {"thread_id": str(thread_id)}}


class MessageResult(NamedTuple):
    """The outcome of processing one user message.

    Attributes:
        response: The reply to send back to the user.
        intent: The intent the orchestrator settled on (for feedback logging).
    """

    response: str
    intent: str


async def process_message(
    orchestrator, message: str, thread_id: str = "cli"
) -> MessageResult:
    """Run a user message through the orchestrator and return the result.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        message: User's message text.
        thread_id: Identifies the conversation. The checkpointer keeps a
            separate memory per thread_id, so pass the Telegram chat_id to give
            each chat its own history. Defaults to a single shared CLI thread.

    Returns:
        A MessageResult with the response text and the classified intent.
    """
    result = await orchestrator.ainvoke(
        # Modality is written on every turn, never left to whatever the last
        # one set. A photo followed by a question would otherwise re-enter the
        # document node with no photo to read.
        {"messages": [HumanMessage(content=message)], "modality": MODALITY_TEXT},
        config=_thread_config(thread_id),
    )
    return MessageResult(response=result["agent_response"], intent=result["intent"])


async def process_document(
    orchestrator, photo: DocumentTurn, thread_id: str = "cli"
) -> MessageResult:
    """Run a photographed document through the orchestrator and return the reply.

    The photo travels as LangGraph runtime context rather than in the state,
    so it is never written to the checkpointer; see DocumentTurn. The graph
    still records the turn, because the node returns the transcript and the
    answer as ordinary messages, which is how a later text follow-up about the
    same document finds its context.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        photo: The image and the caption the user sent with it.
        thread_id: Identifies the conversation, as for process_message().

    Returns:
        A MessageResult with the answer and INTENT_DOCUMENT.
    """
    result = await orchestrator.ainvoke(
        {"modality": MODALITY_DOCUMENT},
        config=_thread_config(thread_id),
        context=photo,
    )
    return MessageResult(response=result["agent_response"], intent=result["intent"])


async def get_history(orchestrator, thread_id) -> list:
    """Return a chat's remembered messages, oldest first.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        The stored messages, or an empty list for a chat that never spoke.
    """
    state = await _read_state(orchestrator, thread_id)
    return state.values.get("messages") or []


async def clear_history(orchestrator, thread_id) -> int:
    """Forget a chat's remembered messages, leaving its preferences alone.

    History and preferences are separate channels in the state, so wiping one
    does not touch the other: someone starting a fresh topic keeps the rules
    they set with /pref. Deletion goes through RemoveMessage rather than
    writing an empty list, because the messages channel appends by design and
    a plain [] would simply add nothing.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        How many messages were forgotten.
    """
    messages = await get_history(orchestrator, thread_id)
    if not messages:
        return 0
    await asyncio.to_thread(
        orchestrator.update_state,
        _thread_config(thread_id),
        {"messages": [RemoveMessage(id=message.id) for message in messages]},
    )
    return len(messages)


async def _read_state(orchestrator, thread_id):
    """Read a chat's stored state without blocking the event loop.

    The graph's synchronous get_state is used rather than aget_state because
    the two disagree: the async path raises "Ambiguous update" once a thread
    has been written to twice outside a graph run, which is exactly what the
    /pref helpers do. A disk-backed checkpointer refuses sync calls made from
    the loop's own thread, so the call is handed to a worker thread instead.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        The graph's StateSnapshot for that conversation.
    """
    return await asyncio.to_thread(orchestrator.get_state, _thread_config(thread_id))


async def _write_preferences(orchestrator, thread_id, instructions: list) -> None:
    """Replace a chat's stored instruction list, off the event loop thread.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
        instructions: The full list to store, which the merge_preferences
            reducer then de-duplicates and caps.
    """
    await asyncio.to_thread(
        orchestrator.update_state,
        _thread_config(thread_id),
        {"preferences": {"instructions": instructions}},
    )


async def get_preferences(orchestrator, thread_id) -> list:
    """Return the standing instructions saved for one chat.

    Reads the persisted checkpointer state directly, without running the graph.
    A chat that has never stored a preference (or never spoken at all) yields an
    empty list.

    Async, and running the read on a worker thread, because a disk-backed
    checkpointer refuses synchronous calls made from the thread that runs the
    event loop. See _read_state for why the sync call is kept at all.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        The list of instruction strings, oldest first.
    """
    state = await _read_state(orchestrator, thread_id)
    return (state.values.get("preferences") or {}).get("instructions") or []


async def add_preference(orchestrator, thread_id, rule: str) -> list:
    """Add one standing instruction to a chat's preferences and return them all.

    Writes to the checkpointer via aupdate_state, so the new rule lands in the
    same `preferences` channel handle_translation reads. The merge_preferences
    reducer de-duplicates and caps the list, so re-adding an existing rule is a
    no-op.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
        rule: The instruction to store (e.g. "Write Serbian in Cyrillic.").

    Returns:
        The updated list of stored instructions.
    """
    updated = (await get_preferences(orchestrator, thread_id)) + [rule]
    await _write_preferences(orchestrator, thread_id, updated)
    return await get_preferences(orchestrator, thread_id)


async def clear_preferences(orchestrator, thread_id) -> None:
    """Remove every standing instruction stored for one chat.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
    """
    await _write_preferences(orchestrator, thread_id, [])


async def remove_preference(orchestrator, thread_id, index: int) -> list:
    """Remove the rule at a 0-based index and return the remaining rules.

    An out-of-range index leaves the stored list unchanged.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
        index: 0-based position of the rule to drop.

    Returns:
        The remaining stored instructions.
    """
    rules = await get_preferences(orchestrator, thread_id)
    if index < 0 or index >= len(rules):
        return rules
    remaining = rules[:index] + rules[index + 1 :]
    await _write_preferences(orchestrator, thread_id, remaining)
    return await get_preferences(orchestrator, thread_id)


async def tidy_preferences(orchestrator, tidier, thread_id) -> list:
    """Merge semantically-duplicate preferences on demand and return the result.

    Reads the stored rules, asks the tidier chain to merge equivalent ones, and
    writes the cleaned list back. A list of fewer than two rules (nothing to
    merge) is returned unchanged without calling the LLM; if the LLM returns
    nothing usable, the original list is kept.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        tidier: The chain from build_preference_tidier().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        The stored instructions after tidying.
    """
    rules = await get_preferences(orchestrator, thread_id)
    if len(rules) < 2:
        return rules
    current = "\n".join(f"- {rule}" for rule in rules)
    cleaned = _parse_rule_lines(await tidier.ainvoke({"current": current}))
    if not cleaned:
        return rules
    await _write_preferences(orchestrator, thread_id, cleaned)
    return await get_preferences(orchestrator, thread_id)
