import re
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agents.intent_classifier import classify, load_classifier
from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

INTENT_KNOWLEDGE_QUESTION = "knowledge_question"
INTENT_TRANSLATION = "translation"
INTENT_OUT_OF_SCOPE = "out_of_scope"

CONFIDENCE_THRESHOLD = 0.6
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
        intent: Classified intent (knowledge_question, translation, out_of_scope).
        agent_response: This turn's reply to send back to the user.
        preferences: Per-chat standing instructions the user has set via the
            /pref command (e.g. {"instructions": ["Reply in Cyrillic."]});
            persisted by the checkpointer and read by handle_translation.
    """

    messages: Annotated[list, append_capped_messages]
    intent: str
    agent_response: str
    preferences: Annotated[dict, merge_preferences]


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
    """
    instructions = (preferences or {}).get("instructions") or []
    if not instructions:
        return "(no special preferences)"
    lines = "\n".join(f"- {instruction}" for instruction in instructions)
    return (
        "Apply these standing user preferences to every reply, whatever the "
        "target language:\n" + lines
    )


def build_orchestrator(rag_chain, translation_chain):
    """Build and compile the LangGraph orchestrator.

    Intent is classified by the local DNN classifier; low-confidence messages
    fall back to the LLM classifier. Routes to the RAG chain, the translation
    chain, or an out-of-scope rejection.

    Args:
        rag_chain: Compiled RAG chain from rag_agent.build_rag_chain().
        translation_chain: Compiled translation chain from
            translation_agent.build_translation_chain().

    Returns:
        A compiled LangGraph that accepts OrchestratorState and returns the final state.
    """
    classifier = load_classifier()
    classifier_chain = _build_classifier_chain()

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
        response = await rag_chain.ainvoke(message)
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

    graph = StateGraph(OrchestratorState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_knowledge_question", handle_knowledge_question)
    graph.add_node("handle_translation", handle_translation)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_to_agent)
    graph.add_edge("handle_knowledge_question", END)
    graph.add_edge("handle_translation", END)
    graph.add_edge("handle_out_of_scope", END)

    return graph.compile(checkpointer=MemorySaver())


def _thread_config(thread_id) -> dict:
    """Build the checkpointer config that scopes state to one conversation."""
    return {"configurable": {"thread_id": str(thread_id)}}


async def process_message(orchestrator, message: str, thread_id: str = "cli") -> str:
    """Run a user message through the orchestrator and return the response.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        message: User's message text.
        thread_id: Identifies the conversation. The checkpointer keeps a
            separate memory per thread_id, so pass the Telegram chat_id to give
            each chat its own history. Defaults to a single shared CLI thread.

    Returns:
        The agent's response string.
    """
    result = await orchestrator.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=_thread_config(thread_id),
    )
    return result["agent_response"]


def get_preferences(orchestrator, thread_id) -> list:
    """Return the standing instructions saved for one chat.

    Reads the persisted checkpointer state directly, without running the graph.
    A chat that has never stored a preference (or never spoken at all) yields an
    empty list.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).

    Returns:
        The list of instruction strings, oldest first.
    """
    state = orchestrator.get_state(_thread_config(thread_id))
    return (state.values.get("preferences") or {}).get("instructions") or []


def add_preference(orchestrator, thread_id, rule: str) -> list:
    """Add one standing instruction to a chat's preferences and return them all.

    Writes to the checkpointer via update_state, so the new rule lands in the
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
    updated = get_preferences(orchestrator, thread_id) + [rule]
    orchestrator.update_state(
        _thread_config(thread_id), {"preferences": {"instructions": updated}}
    )
    return get_preferences(orchestrator, thread_id)


def clear_preferences(orchestrator, thread_id) -> None:
    """Remove every standing instruction stored for one chat.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
    """
    orchestrator.update_state(
        _thread_config(thread_id), {"preferences": {"instructions": []}}
    )


def remove_preference(orchestrator, thread_id, index: int) -> list:
    """Remove the rule at a 0-based index and return the remaining rules.

    An out-of-range index leaves the stored list unchanged.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        thread_id: The conversation id (Telegram chat_id).
        index: 0-based position of the rule to drop.

    Returns:
        The remaining stored instructions.
    """
    rules = get_preferences(orchestrator, thread_id)
    if index < 0 or index >= len(rules):
        return rules
    remaining = rules[:index] + rules[index + 1 :]
    orchestrator.update_state(
        _thread_config(thread_id), {"preferences": {"instructions": remaining}}
    )
    return get_preferences(orchestrator, thread_id)


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
    rules = get_preferences(orchestrator, thread_id)
    if len(rules) < 2:
        return rules
    current = "\n".join(f"- {rule}" for rule in rules)
    cleaned = _parse_rule_lines(await tidier.ainvoke({"current": current}))
    if not cleaned:
        return rules
    orchestrator.update_state(
        _thread_config(thread_id), {"preferences": {"instructions": cleaned}}
    )
    return get_preferences(orchestrator, thread_id)
