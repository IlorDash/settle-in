"""Evals for the translation agent, run against the real OpenAI API.

Every test in tests/unit and tests/integration mocks the LLM, so they check
wiring: that a directive reaches the chain, that a reducer dedupes. None of
them can catch the agent answering in the wrong alphabet or handing back the
Russian source word untranslated, because no model is involved.

These tests close that gap. They cost money and are therefore opt-in: the
"not eval" default in pyproject.toml excludes them from `pytest tests/`.

Run them with:

    .venv/Scripts/python.exe -m pytest -m eval tests/evals -v

The first five cases reproduce bugs that actually reached the CLI. The rest
pin promises SYSTEM_PROMPT makes, so a prompt edit cannot quietly drop one.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.orchestrator import _preferences_directive
from src.agents.translation_agent import build_translation_chain
from src.config import settings

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not settings.openai_api_key,
        reason="needs a real OPENAI_API_KEY",
    ),
]

NO_PREFERENCES: dict = {}
SERBIAN_CYRILLIC_PREFERENCE = {
    "instructions": ["Write Serbian translations in Cyrillic"]
}
EXAMPLES_PREFERENCE = {
    "instructions": ["Provide 5 examples of the use of the word with the translation"]
}

ASK_PARROT_IN_SERBIAN = "Переведи слово попугай на сербский"
ASK_THANK_YOU_IN_SERBIAN = "Как будет спасибо по-сербски"

# Serbian Cyrillic, which unlike Russian has ђ, ј, љ, њ, ћ, џ and no й.
SERBIAN_CYRILLIC_LETTERS = set("абвгдђежзијклљмнњопрстћуфхцчџш")

RUSSIAN_FOR_PARROT = "попугай"
SERBIAN_FOR_PARROT = "папагај"
SERBIAN_FOR_THANK_YOU = "hvala"
# Opens a Serbian rendering of "how do you say"; its presence in a reply means
# the question was translated instead of answered.
SERBIAN_FOR_HOW = "kako"


_REPLIES: dict[tuple, str] = {}


async def translate(
    user_message: str, preferences: dict, history: list | None = None
) -> str:
    """Send one message through the real translation chain.

    Replies are cached per distinct request for the length of the run, so
    several tests can each assert one thing about the same reply without
    paying for the same API call more than once.

    Args:
        user_message: The text the user would type into the bot.
        preferences: Stored preferences, in the shape held in graph state.
        history: Earlier turns as LangChain messages, for follow-up requests
            that only make sense in context. Defaults to no history.

    Returns:
        The agent's reply.
    """
    directive = _preferences_directive(preferences)
    earlier_turns = tuple(message.content for message in history or [])
    request = (user_message, directive, earlier_turns)

    if request not in _REPLIES:
        chain = build_translation_chain()
        _REPLIES[request] = await chain.ainvoke(
            {
                "input": user_message,
                "history": history or [],
                "preferences": directive,
            }
        )
    return _REPLIES[request]


def is_serbian_cyrillic(text: str) -> bool:
    """Report whether a reply is written in Cyrillic.

    Args:
        text: The agent's reply.

    Returns:
        True when any Serbian Cyrillic letter appears in the text.
    """
    return any(character in SERBIAN_CYRILLIC_LETTERS for character in text.lower())


async def test_explicit_latin_request_overrides_stored_serbian_cyrillic_preference():
    # The bug: a saved "in Cyrillic" rule outranked the message asking for
    # Latin, because a system-message preference outweighs a user request.
    reply = await translate(
        "Переведи слово попугай на сербский латиница", SERBIAN_CYRILLIC_PREFERENCE
    )

    assert not is_serbian_cyrillic(reply), f"asked for Latin, got Cyrillic: {reply!r}"


async def test_stored_preference_still_applies_without_an_explicit_request():
    # The override above must be scoped to the one reply. A message that says
    # nothing about script must still follow the stored Cyrillic rule.
    reply = await translate(ASK_PARROT_IN_SERBIAN, SERBIAN_CYRILLIC_PREFERENCE)

    assert is_serbian_cyrillic(reply), f"preference ignored, got Latin: {reply!r}"


async def test_translation_does_not_echo_the_russian_source_word():
    # The worst bug of the set: the agent answered "попугай", the Russian word
    # it was asked to translate, instead of the Serbian "папагај".
    reply = await translate(ASK_PARROT_IN_SERBIAN, SERBIAN_CYRILLIC_PREFERENCE)

    assert RUSSIAN_FOR_PARROT not in reply.lower(), f"echoed the source: {reply!r}"


async def test_translation_is_actually_correct():
    # Guards the opposite failure: a reply can avoid the source word and still
    # be wrong, so pin the expected Serbian translation.
    reply = await translate(ASK_PARROT_IN_SERBIAN, SERBIAN_CYRILLIC_PREFERENCE)

    assert SERBIAN_FOR_PARROT in reply.lower(), f"wrong translation: {reply!r}"


async def test_examples_preference_applies_to_a_non_serbian_target():
    # The bug: an "add 5 examples" rule was obeyed for Serbian but silently
    # dropped when translating into Russian, because the other stored rule
    # mentioned Serbian and biased the whole set.
    reply = await translate("Translate a plate to russian", EXAMPLES_PREFERENCE)

    assert len(reply.splitlines()) > 2, f"examples missing for Russian: {reply!r}"


async def test_reply_is_terse_when_no_preferences_are_stored():
    # SYSTEM_PROMPT promises the translation "and, by default, nothing else".
    # Injecting a preferences block made the agent chatty, answering in whole
    # sentences, so pin the plain default it is supposed to fall back to.
    reply = await translate(ASK_PARROT_IN_SERBIAN, NO_PREFERENCES)

    assert len(reply.split()) <= 3, f"expected a bare translation: {reply!r}"


async def test_serbian_is_written_in_latin_by_default():
    # "use Latin by default": with nothing stored the reply must not be
    # Cyrillic, otherwise the stored-preference tests above prove nothing.
    reply = await translate(ASK_PARROT_IN_SERBIAN, NO_PREFERENCES)

    assert not is_serbian_cyrillic(reply), f"default should be Latin: {reply!r}"


async def test_the_request_itself_is_not_translated():
    # "Do not translate the request itself": the agent must answer the
    # question rather than render it in Serbian, which would open with
    # "kako se kaže". Both halves are needed, because the absence of "kako"
    # alone would also hold for an empty or failed reply.
    reply = await translate(ASK_THANK_YOU_IN_SERBIAN, NO_PREFERENCES)

    assert SERBIAN_FOR_THANK_YOU in reply.lower(), f"word missing: {reply!r}"
    assert SERBIAN_FOR_HOW not in reply.lower(), f"translated the question: {reply!r}"


async def test_asking_to_always_do_something_points_at_the_pref_command():
    # The agent cannot save settings. It used to promise it would remember a
    # rule, which was a lie; it must redirect to the real mechanism instead.
    reply = await translate(
        "Always write Serbian translations in Cyrillic from now on", NO_PREFERENCES
    )

    assert "/pref" in reply, f"should redirect to /pref: {reply!r}"


async def test_a_followup_is_resolved_from_the_conversation_history():
    # "а латиницей?" means nothing on its own. The agent has to read the
    # previous turn to know it should repeat that translation in Latin, which
    # is the whole point of passing history into the prompt.
    history = [
        HumanMessage(content=ASK_PARROT_IN_SERBIAN),
        AIMessage(content=SERBIAN_FOR_PARROT),
    ]

    reply = await translate("а латиницей?", NO_PREFERENCES, history)

    assert not is_serbian_cyrillic(reply), f"follow-up not resolved: {reply!r}"
