from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents.intent_classifier import classify, load_classifier
from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

INTENT_KNOWLEDGE_QUESTION = "knowledge_question"
INTENT_TRANSLATION = "translation"
INTENT_OUT_OF_SCOPE = "out_of_scope"

CONFIDENCE_THRESHOLD = 0.6

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about living in Serbia and with "
    "Serbian-English and Serbian-Russian translation. Please rephrase your "
    "message around one of those."
)

CLASSIFICATION_PROMPT = (
    "You are a message classifier for an immigrant assistance bot in Serbia. "
    "Classify the user's message into one of two categories:\n\n"
    f"- {INTENT_KNOWLEDGE_QUESTION}: Questions about living in Serbia, "
    "immigration procedures, documents, laws, banks, taxes, healthcare, etc.\n"
    f"- {INTENT_TRANSLATION}: Requests to translate text between Serbian and English, "
    "or questions about how to say something in Serbian/English.\n\n"
    "Respond with ONLY the category name, nothing else."
)


class OrchestratorState(TypedDict):
    """State that flows through the orchestrator graph.

    Attributes:
        user_message: The original message from the user.
        intent: Classified intent (knowledge_question, translation, out_of_scope).
        agent_response: Final response to send back to the user.
    """

    user_message: str
    intent: str
    agent_response: str


def _build_classifier_chain():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFICATION_PROMPT),
            ("human", "{input}"),
        ]
    )

    return prompt | llm | StrOutputParser()


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
        intent, confidence = classify(classifier, state["user_message"])
        if confidence >= CONFIDENCE_THRESHOLD:
            return {"intent": intent}

        # Low confidence: ask the LLM classifier for a second opinion.
        result = await classifier_chain.ainvoke({"input": state["user_message"]})
        llm_intent = result.strip().lower()
        if INTENT_TRANSLATION in llm_intent:
            return {"intent": INTENT_TRANSLATION}
        return {"intent": INTENT_KNOWLEDGE_QUESTION}

    async def handle_knowledge_question(state: OrchestratorState) -> dict:
        response = await rag_chain.ainvoke(state["user_message"])
        return {"agent_response": response}

    async def handle_translation(state: OrchestratorState) -> dict:
        response = await translation_chain.ainvoke({"input": state["user_message"]})
        return {"agent_response": response}

    async def handle_out_of_scope(state: OrchestratorState) -> dict:
        return {"agent_response": OUT_OF_SCOPE_MESSAGE}

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

    return graph.compile()


async def process_message(orchestrator, message: str) -> str:
    """Run a user message through the orchestrator and return the response.

    Args:
        orchestrator: Compiled orchestrator graph from build_orchestrator().
        message: User's message text.

    Returns:
        The agent's response string.
    """
    result = await orchestrator.ainvoke({"user_message": message})
    return result["agent_response"]
