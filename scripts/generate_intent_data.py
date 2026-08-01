"""Generate labeled training data for SettleIn's intent classifier.

Uses GPT-4o-mini as a "teacher" to synthesize example queries for each intent
(knowledge distillation): the LLM that currently classifies intents produces the
labeled data our small, free classifier will learn from. This is a one-time,
offline step -- run it whenever you want to (re)build the training set.
"""

import csv
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import settings

OUTPUT_PATH = Path("data/intent_training.csv")
EXAMPLES_PER_INTENT = 250
BATCH_SIZE = 30

# One generation prompt per intent. {n} is filled in per batch.
# SettleIn's users are mostly Russian-speaking immigrants in Serbia, so real
# messages are mostly in Russian, with Serbian and some English mixed in.
# The training data must match that language mix or TF-IDF misses real input.
INTENT_PROMPTS = {
    "knowledge_question": (
        "Generate {n} short, varied questions a Russian-speaking immigrant in Serbia would "
        "ask about living there: residency (боравак), the white card (бели картон), opening a "
        "bank account, health insurance, taxes and PIB, utility bills, the visa regime, and "
        "e-government services. Write approx half of them in Russian (the way real users ask), "
        "and half in English. One question per line, no numbering, no quotes."
    ),
    "translation": (
        "Generate {n} short, varied translation requests a Russian-speaking immigrant in Serbia "
        "would send. IMPORTANT: make about HALF of them just a pasted Serbian sentence "
        "with NO instruction at all -- the user simply forwards text they want understood "
        "(a chat message, an SMS, an official notice, or a casual sentence). For the other "
        "half, use short instructions like 'Как будет по-сербски ...', 'Переведи на русский ...', "
        "'Проверь: ...', 'Что значит ...', usually followed by a short phrase. Vary the length "
        "and topic (personal, work, scheduling, official). Mix in a few English requests. "
        "One request per line, no numbering, no quotes."
    ),
    "out_of_scope": (
        "Generate {n} short, varied OFF-TOPIC messages, unrelated to living in Serbia or to "
        "translation: weather, sport, cooking recipes, math, jokes, tech support, movies and "
        "music, personal chit-chat, and other countries. Write MOST of them in Russian, with "
        "some in English and Serbian. One message per line, no numbering, no quotes."
    ),
}


def generate_for_intent(llm, prompt_template, total, batch):
    """Ask the LLM for `total` example queries, in batches, deduplicated.

    Args:
        llm: A ChatOpenAI instance.
        prompt_template: The intent's prompt with an {n} placeholder.
        total: How many unique examples we want.
        batch: How many to request per LLM call.

    Returns:
        A list of `total` unique example strings.
    """
    seen = set()
    max_attempts = total // batch + 10  # guard against endless repetition
    attempts = 0
    while len(seen) < total and attempts < max_attempts:
        attempts += 1
        prompt = prompt_template.format(n=min(batch, total - len(seen)))
        response = llm.invoke([HumanMessage(content=prompt)]).content
        for line in response.splitlines():
            cleaned = line.strip().strip('-").').strip()
            if cleaned:
                seen.add(cleaned)
    return list(seen)[:total]


def main() -> None:
    """Generate examples for every intent and write them to a CSV."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9,
        openai_api_key=settings.openai_api_key,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        for intent, template in INTENT_PROMPTS.items():
            examples = generate_for_intent(llm, template, EXAMPLES_PER_INTENT, BATCH_SIZE)
            for text in examples:
                writer.writerow([text, intent])
            print(f"{intent}: {len(examples)} examples")

    print("Saved to", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
