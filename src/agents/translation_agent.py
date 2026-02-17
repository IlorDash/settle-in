from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are a translator between Serbian and English. "
    "Detect the language of the input text and translate it to the other language. "
    "If the text is in Serbian, translate to English. "
    "If the text is in English, translate to Serbian. "
    "Return ONLY the translation, nothing else."
)


def build_translation_chain():
    """Build an LCEL chain that translates between Serbian and English.

    Returns:
        A Runnable chain that accepts a text string and returns the translation.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    return prompt | llm | StrOutputParser()


async def translate(chain, text: str) -> str:
    """Translate text through the translation chain.

    Args:
        chain: Translation chain built by build_translation_chain().
        text: Text to translate (Serbian or English).

    Returns:
        Translated text in the opposite language.
    """
    return await chain.ainvoke({"input": text})
