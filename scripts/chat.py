"""Interactive CLI to test the orchestrator locally, without Telegram.

Type a message and see the classified intent, the DNN confidence, and the
bot's reply. Useful for checking classification and out-of-scope rejection
without touching the deployed bot. Run from the repo root:

    python -m scripts.chat
"""

import asyncio

from src.agents.intent_classifier import classify, load_classifier
from src.agents.orchestrator import build_orchestrator, process_message
from src.agents.rag_agent import build_rag_chain
from src.agents.translation_agent import build_translation_chain
from src.knowledge.vectorstore import get_retriever, load_vectorstore


async def main() -> None:
    retriever = get_retriever(load_vectorstore())
    orchestrator = build_orchestrator(
        build_rag_chain(retriever), build_translation_chain()
    )
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

        intent, confidence = classify(classifier, message)
        print(f"[dnn intent={intent} confidence={confidence:.2f}]")
        response = await process_message(orchestrator, message)
        print(f"bot> {response}")


if __name__ == "__main__":
    asyncio.run(main())
