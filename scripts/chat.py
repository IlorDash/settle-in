"""Interactive CLI to test the orchestrator locally, without Telegram.

Type a message and see the classified intent, the DNN confidence, and the
bot's reply. Useful for checking classification and out-of-scope rejection
without touching the deployed bot. Run from the repo root:

    python -m scripts.chat
"""

import asyncio

from src.agents.intent_classifier import classify, load_classifier
from src.agents.orchestrator import (
    add_preference,
    build_orchestrator,
    build_preference_tidier,
    clear_preferences,
    get_preferences,
    process_message,
    remove_preference,
    tidy_preferences,
)
from src.agents.rag_agent import build_rag_chain
from src.agents.translation_agent import build_translation_chain
from src.knowledge.vectorstore import get_retriever, load_vectorstore

CLI_THREAD = "cli"


def _print_prefs(rules: list) -> None:
    """Print the preference list as a 1-based numbered block (or '(none)')."""
    if not rules:
        print("prefs> (none)")
        return
    print("prefs>")
    for position, rule in enumerate(rules, 1):
        print(f"  {position}. {rule}")


async def handle_pref_command(orchestrator, tidier, message: str) -> None:
    """Mirror the Telegram /pref command in the CLI for quick local testing.

    Lets you set, prune, tidy, and inspect standing preferences on the shared
    "cli" thread, the same one process_message uses, so you can check that a
    saved rule reaches the translation agent.
    """
    args = message.split()[1:]
    if not args:
        _print_prefs(get_preferences(orchestrator, CLI_THREAD))
        return
    action = args[0].lower()
    if action == "add":
        rule = " ".join(args[1:]).strip()
        if not rule:
            print("prefs> usage: /pref add <rule>")
            return
        _print_prefs(add_preference(orchestrator, CLI_THREAD, rule))
        return
    if action == "remove" and len(args) > 1 and args[1].isdigit():
        _print_prefs(remove_preference(orchestrator, CLI_THREAD, int(args[1]) - 1))
        return
    if action == "tidy":
        _print_prefs(await tidy_preferences(orchestrator, tidier, CLI_THREAD))
        return
    if action == "clear":
        clear_preferences(orchestrator, CLI_THREAD)
        print("prefs> cleared")
        return
    print("prefs> usage: /pref [add <rule> | remove <number> | tidy | clear]")


async def main() -> None:
    retriever = get_retriever(load_vectorstore())
    orchestrator = build_orchestrator(
        build_rag_chain(retriever), build_translation_chain()
    )
    tidier = build_preference_tidier()
    classifier = load_classifier()

    print("SettleIn CLI - type a message, Ctrl+C to quit.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not message:
            continue

        if message.startswith("/pref"):
            await handle_pref_command(orchestrator, tidier, message)
            continue

        intent, confidence = classify(classifier, message)
        print(f"[dnn intent={intent} confidence={confidence:.2f}]")
        result = await process_message(orchestrator, message)
        print(f"bot [routed={result.intent}]> {result.response}")


if __name__ == "__main__":
    asyncio.run(main())
