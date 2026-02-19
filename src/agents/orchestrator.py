from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

INTENT_KNOWLEDGE_QUESTION = "knowledge_question"
INTENT_TRANSLATION = "translation"

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
        intent: Classified intent (knowledge_question or translation).
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

    Creates a StateGraph with three nodes:
    1. classify_intent — LLM classifies the user's message
    2. handle_knowledge_question — calls the RAG chain
    3. handle_translation — calls the translation chain

    A conditional edge routes from classify_intent to the appropriate handler.

    Args:
        rag_chain: Compiled RAG chain from rag_agent.build_rag_chain().
        translation_chain: Compiled translation chain from
            translation_agent.build_translation_chain().

    Returns:
        A compiled LangGraph that accepts OrchestratorState and returns the final state.
    """
    classifier_chain = _build_classifier_chain()

    async def classify_intent(state: OrchestratorState) -> dict:
        result = await classifier_chain.ainvoke({"input": state["user_message"]})
        intent = result.strip().lower()

        if INTENT_TRANSLATION in intent:
            return {"intent": INTENT_TRANSLATION}
        return {"intent": INTENT_KNOWLEDGE_QUESTION}

    async def handle_knowledge_question(state: OrchestratorState) -> dict:
        response = await rag_chain.ainvoke(state["user_message"])
        return {"agent_response": response}

    async def handle_translation(state: OrchestratorState) -> dict:
        response = await translation_chain.ainvoke({"input": state["user_message"]})
        return {"agent_response": response}

    def route_to_agent(
        state: OrchestratorState,
    ) -> Literal["handle_knowledge_question", "handle_translation"]:
        if state["intent"] == INTENT_TRANSLATION:
            return "handle_translation"
        return "handle_knowledge_question"

    graph = StateGraph(OrchestratorState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_knowledge_question", handle_knowledge_question)
    graph.add_node("handle_translation", handle_translation)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_to_agent)
    graph.add_edge("handle_knowledge_question", END)
    graph.add_edge("handle_translation", END)

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
