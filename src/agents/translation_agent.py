from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are a translation assistant for immigrants in Serbia. Users write in "
    "Russian, Serbian, or English. A message may be text to translate, or a "
    "request such as 'Как будет X по-сербски', 'how do you say X in English', "
    "or 'переведи на русский: ...'. Work out what the user wants translated and "
    "into which language, then reply with ONLY that translation. Add the Latin "
    "transliteration in parentheses ONLY when the result is Serbian - never for "
    "Russian (which has no Latin form) or English. If the target "
    "language is not stated, translate Serbian and English to each other, or "
    "into the user's own language. Do not translate the request itself or add "
    "explanations."
)


def build_translation_chain():
    """Build an LCEL chain that translates between Russian, Serbian, and English.

    Returns:
        A Runnable chain that accepts a text string and returns the translation.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    return prompt | llm | StrOutputParser()


async def translate(chain, text: str) -> str:
    """Translate text through the translation chain.

    Args:
        chain: Translation chain built by build_translation_chain().
        text: Text to translate, or a translation request, in Russian,
            Serbian, or English.

    Returns:
        The translation the user asked for.
    """
    return await chain.ainvoke({"input": text})
