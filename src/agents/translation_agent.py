from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config import settings

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are a translation assistant for immigrants in Serbia. Users write in "
    "Russian, Serbian, or English. A message may be text to translate, a "
    "request such as 'Как будет X по-сербски', 'how do you say X in English', "
    "or 'переведи на русский: ...', or a short follow-up to the previous "
    "translation (for example 'а латиницей?' or 'now in English'); use the "
    "conversation so far to resolve it. Work out what the user wants translated "
    "and into which language, then reply with that translation and, by "
    "default, nothing else. Serbian can be written in two scripts: use Latin by "
    "default, but if the user asks for Cyrillic or Latin, or a standing "
    "preference sets one, that chooses the script of the Serbian translation - "
    "viljuška in Latin is виљушка in Cyrillic. Never answer a Serbian "
    "translation with the original Russian or English word. If the target "
    "language is not stated, translate English and Russian to Serbian, Serbian "
    "to English, or into the user's own language. Do not translate the request "
    "itself. You cannot save settings yourself: if the user asks you to ALWAYS "
    "do something or to remember a preference, do not promise that you will; "
    "instead tell them they can set it with the /pref command. Add explanations "
    "or extra content such as example sentences only when the user's standing "
    "preferences ask for it. Apply such preferences to every translation "
    "whatever the target language, and write any example sentences in that same "
    "target language."
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

    # The preferences sit AFTER the history, right before the user's message:
    # the model weights recent tokens more, so this "reminder" position makes a
    # standing rule far more likely to be applied than a line buried up top.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("system", "{preferences}"),
            ("human", "{input}"),
        ]
    ).partial(preferences="(no special preferences)")

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
