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
EXAMPLES_PER_INTENT = 150
BATCH_SIZE = 30

# One generation prompt per intent. {n} is filled in per batch.
INTENT_PROMPTS = {
    "knowledge_question": (
        "Generate {n} short, varied questions an immigrant in Serbia might ask about "
        "living there: residency permits, the white card, opening a bank account, health "
        "insurance, taxes and PIB, utility bills, the visa regime, and e-government "
        "services. One question per line, no numbering, no quotes."
    ),
    "translation": (
        "Generate {n} short, varied requests to translate text between Serbian and "
        "English. Mix both directions and phrasings (for example 'translate ... to "
        "Serbian', 'how do I say ... in English', 'what does ... mean'). One request per "
        "line, no numbering, no quotes."
    ),
    "out_of_scope": (
        "Generate {n} short, varied user messages that are OFF-TOPIC for an assistant "
        "that only helps with (a) living in Serbia as an immigrant and (b) Serbian-English "
        "translation. Cover diverse unrelated topics: weather, sports, cooking recipes, "
        "math problems, jokes, general tech support, movies and music, personal chit-chat, "
        "and questions about other countries. One message per line, no numbering, no quotes."
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
