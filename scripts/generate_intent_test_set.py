"""Generate the labelled test set the routing evaluation is scored on.

This is NOT the training-data generator. scripts/generate_intent_data.py
writes what the local classifier learns from; this writes what both
classifiers are then scored on, and the two must not be produced the same way.

Three things keep the comparison fair:

  * The wording is seeded from real messages, not invented from a topic list.
    A set generated the way the training set was would share the distribution
    the local classifier was fitted to, and the language model it is measured
    against has no such advantage.
  * The generator is a different model from the one that classifies. The
    escalation classifier runs gpt-4o-mini, so scoring it on gpt-4o-mini prose
    would hand it its own idiom.
  * Every cell of the intent-by-language grid is generated on its own, with
    its own prompt and its own quota. One prompt asked for a language mix
    returns whatever mix the exemplars happened to carry, which is not a
    balance and cannot be reported per language. The grid is not square -
    see GRID for why Serbian appears under one intent only.

The real messages are shown as style exemplars only. Their labels are never
used: only the translation labels in the real set are reliable, and the sample
is drawn across the whole file, so it is asked for form and not for content.
Because those exemplars carry an intent of their own, every generated line is
also screened against the intent it was asked for - see REJECTIONS.

No real message reaches the output, and the output carries no personal
content, so it belongs in git and the thesis numbers stay reproducible.
Run from the repo root:

    python -m scripts.generate_intent_test_set
    python -m scripts.generate_intent_test_set --verify

The second form writes nothing. It reads the file back and puts every row
through the same screens that admitted it, plus the grid quotas and a
pairwise duplicate check, and exits non-zero if anything fails. A screen that
is tightened afterwards makes the file stale, and this is what says so.

Every label it writes still has to be read by hand before anything is scored.
"""

import collections
import csv
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_OUT_OF_SCOPE,
    INTENT_TRANSLATION,
)
from src.config import settings

OUTPUT_PATH = Path("data/intent_test.csv")
TRAINING_PATH = Path("data/intent_training.csv")
REAL_PATH = Path(os.environ.get("SETTLEIN_REAL_DATA", "data/intent_training_real.csv"))


LANGUAGES = {"ru": "Russian", "sr": "Serbian", "en": "English"}
BATCH_SIZE = 25

# The grid, 150 messages per intent. It is deliberately not square. This
# system serves Russian-speaking newcomers, so Serbian reaches it as text
# somebody else wrote and the user wants understood, not as a question the
# user is asking. The training data already works that way: of its 250
# knowledge questions not one is written in Serbian, and its Serbian messages
# sit almost entirely under translation. Serbian is a language this system
# reads, not one it is addressed in, so the two other intents are split
# between Russian and English.
GRID = {
    INTENT_KNOWLEDGE_QUESTION: {"ru": 75, "en": 75},
    INTENT_TRANSLATION: {"ru": 50, "sr": 50, "en": 50},
    INTENT_OUT_OF_SCOPE: {"ru": 75, "en": 75},
}

# Deliberately not gpt-4o-mini: that is what the escalation classifier runs,
# and a classifier should not be scored on text its own model wrote. The
# temperature sits well below 1.0 because the first run at 1.0 produced
# ungrammatical Serbian ("Per si slobodan", "Kako bude po-serbski") while
# imitating the exemplars' informal spelling.
GENERATOR_MODEL = "gpt-5.4"
GENERATOR_TEMPERATURE = 0.6

# How many real messages are shown as style exemplars, and the seed that picks
# them. The seed is fixed so a re-run shows the generator the same sample.
STYLE_SAMPLE_SIZE = 40
STYLE_SAMPLE_SEED = 20260826

# Two messages count as near duplicates when this share of their words is
# shared. Set by hand: 0.8 catches a reworded copy while leaving two genuinely
# different questions about the same procedure alone.
DUPLICATE_OVERLAP = 0.8

# A candidate that sits inside a known message, or swallows one whole, is a
# copy even when the overlap score is low. Short fragments are exempt: an
# out-of-scope message is often one word, and one word turns up inside almost
# anything.
CONTAINMENT_MIN_WORDS = 4

# Containment alone is not enough. Several real messages are a bare
# instruction - "как будет по сербски" - and every Russian translation request
# has to contain those words, so containment on its own rejected 13 correct
# messages. The pair must also share this much of their vocabulary, which a
# long request built on a short stub does not.
CONTAINMENT_MIN_OVERLAP = 0.6

# The longest real message is well under this. A candidate above it is the
# model running on rather than a message somebody would send.
MAX_MESSAGE_CHARS = 200

# Below this length a message carries too few words to judge by vocabulary, so
# the language check stands aside and the hand review decides.
LANGUAGE_CHECK_MIN_WORDS = 4

# Latin, Cyrillic, punctuation, currency and emoji. Anything else means the
# model slipped scripts mid-sentence, which happened once in the first run.
ALLOWED_CHARACTERS = re.compile(r"^[\s\x20-ɏͰ-ӿ -⁯₠-₿" r"☀-➿\U0001f300-\U0001faff]*$")

# Words that place a Latin-script message in Serbian or in English. The two
# sets are kept disjoint on purpose: "a", "i", "to", "do" and "me" are
# ordinary words in both languages and would vote for whichever list held
# them.
LANGUAGE_MARKERS = {
    "sr": {
        "da",
        "li",
        "je",
        "su",
        "sam",
        "nije",
        "nema",
        "ima",
        "ne",
        "na",
        "za",
        "sa",
        "se",
        "u",
        "ili",
        "koji",
        "kako",
        "koliko",
        "gde",
        "kada",
        "kad",
        "sta",
        "mozes",
        "moze",
        "treba",
        "ovde",
        "ovo",
        "jos",
        "vec",
        "ali",
        "pa",
        "od",
        "kod",
        "posle",
        "pre",
        "neki",
        "jel",
        "bez",
        "sada",
        "danas",
        "sutra",
        "hocu",
        "molim",
        "hvala",
    },
    "en": {
        "the",
        "is",
        "are",
        "was",
        "were",
        "does",
        "did",
        "can",
        "could",
        "you",
        "your",
        "my",
        "for",
        "it",
        "what",
        "how",
        "and",
        "of",
        "have",
        "has",
        "need",
        "with",
        "about",
        "there",
        "here",
        "they",
        "this",
        "that",
        "will",
        "would",
        "should",
        "please",
        "thanks",
    },
}

# A knowledge question wrapped in "translate this" is a translation request,
# and the first run produced 30 of them because the exemplars are mostly
# translation requests. They are rejected rather than relabelled: the label
# has to come from the prompt, not from a guess made after the fact.
TRANSLATION_CUES = re.compile(
    r"(переведи|как будет по|как сказать по|что значит|что означает|проверь"
    r"|можешь перевести|на сербский|по-сербски|по сербски"
    r"|prevedi|prevesti|kako se kaze|kako da kazem|kako bude|na srpski"
    r"|proveri|sta znaci|translate|what does|how do you say)",
    re.IGNORECASE,
)

STYLE_PREAMBLE = (
    "Below are real messages one Russian-speaking immigrant in Serbia sent to "
    "a Telegram assistant. Study only HOW they are written: how short they "
    "are, how blunt, how little punctuation they use, how rarely they say "
    "please. Ignore what they are about, and ignore which languages they are "
    "in - the language of what you write is fixed separately below.\n\n"
    "{examples}\n\n"
)

LANGUAGE_RULE = (
    " Write every message in {language}. It must be correct, natural "
    "{language} that a person would actually type - informal is right, "
    "ungrammatical is not. Serbian may be written in Latin script without "
    "diacritics, which is how people type it in chat."
)

FORMAT_RULE = (
    " One message per line, at most 25 words each. No numbering, no bullets, "
    "no surrounding quotation marks, no explanation, no blank lines. Do not "
    "reuse a sentence from the examples above."
)

KNOWLEDGE_PROMPT = (
    "Write {n} new messages, each one a direct question about settling in "
    "Serbia: the residence permit (boravak), the white card (beli karton), "
    "opening a bank account, health insurance, taxes and PIB, utility bills, "
    "the visa regime, or e-government services. Each message must be the "
    "question itself and nothing else. Never ask for a translation, never ask "
    "anyone to check or rephrase wording, and never quote a sentence to be "
    "translated - those belong to a different category and are rejected."
)

OUT_OF_SCOPE_PROMPT = (
    "Write {n} new messages that are off topic: nothing to do with living in "
    "Serbia, and nothing to translate. Cover weather, sport, cooking, "
    "arithmetic, jokes, films, gadgets and small talk. Make about a fifth of "
    "them a bare one-word or two-word fragment carrying no request at all."
)

# Translation needs a prompt per language, because the request and the text it
# carries have to be in different languages. The first run asked in Serbian
# for a translation into Serbian, which is not a request anybody would send.
TRANSLATION_PROMPTS = {
    "ru": (
        "Write {n} new messages, each one a translation request written in "
        "Russian. Use two kinds in roughly equal numbers. First: a short "
        "Russian instruction followed by a RUSSIAN phrase to put into Serbian "
        "- for example 'Переведи на сербский' or 'Как будет по-сербски' plus "
        "the phrase. Second: a short Russian instruction followed by a "
        "SERBIAN phrase to be understood - for example 'Что значит' or "
        "'Переведи на русский' plus the phrase. The instruction and the "
        "phrase must never be in the same language."
    ),
    "sr": (
        "Write {n} new messages, each one a translation request written in "
        "Serbian. Use two kinds in roughly equal numbers. First: a short "
        "Serbian instruction followed by a SERBIAN phrase to be put into "
        "Russian or English - for example 'Kako se ovo kaze na ruskom' plus "
        "the phrase. Second: a bare forwarded Serbian sentence with no "
        "instruction at all - an SMS, a chat message or an official notice "
        "the sender simply wants understood. Let several of those "
        "forwarded sentences be about residence permits, banks or "
        "utility bills, since a user is sent such messages and forwards "
        "them as they are. Never ask for a translation into Serbian, "
        "since the request is already in Serbian."
    ),
    "en": (
        "Write {n} new messages, each one a translation request written in "
        "English. Use two kinds in roughly equal numbers. First: an English "
        "question asking how to say an ENGLISH phrase in Serbian - for "
        "example 'How do you say I will be ten minutes late in Serbian'. "
        "Second: an English question asking what a SERBIAN phrase means. The "
        "question and the phrase must never be in the same language."
    ),
}

# Every screen a candidate can fail, so a run reports which one did the work
# rather than a single opaque count.
REJECTIONS = (
    "duplicate of known text",
    "too long",
    "foreign script",
    "reads as a translation request",
    "wrong language",
)


def read_column(path: Path, column: str) -> list[str]:
    """Read one column out of a labelled CSV.

    Args:
        path: The CSV to read. It must have a header row.
        column: Name of the column to collect.

    Returns:
        Every non-empty value in that column, in file order.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        return [row[column].strip() for row in csv.DictReader(handle) if row[column]]


def normalise(text: str) -> str:
    """Fold a message to the form the duplicate check compares."""
    return " ".join(text.lower().split())


def words_of(text: str) -> list[str]:
    """Split a message into lowercase word tokens, punctuation removed."""
    return re.findall(r"\w+", text.lower())


@dataclass
class DuplicateFilter:
    """Rejects a generated message that is too close to one already written.

    Attributes:
        exact: Normalised forms that must not appear in the output at all.
        word_sets: Word sets of the same messages, for the overlap test.
            Punctuation is stripped out of them, because splitting on
            whitespace alone let "Нормально" and "Нормально?" past each other
            as different words.

    Messages already written are added as the run goes on, so the set is also
    screened against itself. Without that the second run produced seven pairs
    differing by one word, such as "буду через 10 минут" and "через 15 минут".
    """

    exact: set[str]
    word_sets: list[frozenset[str]]

    @classmethod
    def over(cls, corpora: list[list[str]]) -> "DuplicateFilter":
        """Build a filter over every message in the given corpora.

        Each message is indexed whole and line by line. Real messages are
        often several lines long, and the generator is asked for one message
        per line, so a copied line would otherwise slip past a filter that
        only knew the whole message.
        """
        messages = {
            folded
            for corpus in corpora
            for text in corpus
            for folded in [normalise(text), *map(normalise, text.splitlines())]
            if folded
        }
        return cls(
            exact=set(messages),
            word_sets=[frozenset(words_of(text)) for text in messages],
        )

    def add(self, text: str) -> None:
        """Take a message into the corpus, so nothing later repeats it."""
        folded = normalise(text)
        self.exact.add(folded)
        self.word_sets.append(frozenset(words_of(folded)))

    def rejects(self, candidate: str) -> bool:
        """Return True when the candidate is a near duplicate of the corpus."""
        folded = normalise(candidate)
        if not folded or folded in self.exact:
            return True
        words = frozenset(words_of(folded))
        return any(self._too_close(words, other) for other in self.word_sets)

    def _too_close(self, words: frozenset[str], other: frozenset[str]) -> bool:
        """Compare one candidate word set against one known message."""
        shared = words & other
        if not shared:
            return False
        overlap = len(shared) / len(words | other)
        if overlap >= DUPLICATE_OVERLAP:
            return True
        smaller = min(len(words), len(other))
        return (
            smaller >= CONTAINMENT_MIN_WORDS
            and len(shared) == smaller
            and overlap >= CONTAINMENT_MIN_OVERLAP
        )


@dataclass(frozen=True)
class Cell:
    """One square of the intent-by-language grid.

    Attributes:
        intent: The label every message in this cell carries.
        language: The two-letter code every message must be written in.
        quota: How many messages this square holds.
    """

    intent: str
    language: str
    quota: int = 50

    def __str__(self) -> str:
        return f"{self.intent}/{self.language}"


@dataclass(frozen=True)
class Generator:
    """What every cell needs: a model, the style sample, and the screens.

    Attributes:
        llm: The generating model.
        style_examples: Real messages shown for their form alone.
        guard: Rejects anything too close to a real or training message.
    """

    llm: ChatOpenAI
    style_examples: tuple[str, ...]
    guard: DuplicateFilter


def wrong_language(candidate: str, cell: "Cell") -> bool:
    """Return True when a message is not in the language it was asked for.

    Russian is separated from the other two by script. Serbian and English
    share the Latin alphabet, so they are told apart by which list of common
    words the message draws on more. A message that draws on neither is left
    alone: the screen cannot tell, and the hand review can.

    The screen sees vocabulary and not grammar, so a message written in
    broken Serbian passes it. Correct Serbian is asked for in the prompt and
    confirmed by the hand review, not here.
    """
    tokens = set(words_of(candidate))
    if len(tokens) < LANGUAGE_CHECK_MIN_WORDS:
        return False
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", candidate))
    latin = len(re.findall(r"[A-Za-z]", candidate))
    if cell.intent == INTENT_TRANSLATION:
        return not carries(candidate, cell.language, tokens)
    if cell.language == "ru":
        return cyrillic <= latin
    if cyrillic > latin:
        return True
    other = "en" if cell.language == "sr" else "sr"
    return len(tokens & LANGUAGE_MARKERS[other]) > len(
        tokens & LANGUAGE_MARKERS[cell.language]
    )


def carries(candidate: str, language: str, tokens: set[str]) -> bool:
    """Return True when a message contains the given language at all.

    Presence, not dominance. A translation request holds a phrase in another
    language on purpose, and "Переведи на русский" followed by a long Serbian
    sentence has more Latin letters than Cyrillic ones while still being a
    Russian request. Testing dominance rejected 50 such messages in the second
    run and left that half of the Russian cell nearly empty.
    """
    if language == "ru":
        return bool(re.search(r"[Ѐ-ӿ]", candidate))
    return bool(tokens & LANGUAGE_MARKERS[language])


def rejection_reason(candidate: str, cell: Cell, guard: DuplicateFilter) -> str | None:
    """Name the screen a candidate fails, or None when it passes them all.

    Args:
        candidate: One generated message.
        cell: The square of the grid the message was generated for.
        guard: Rejects anything too close to a real or training message.

    Returns:
        One of REJECTIONS, or None.
    """
    if len(candidate) > MAX_MESSAGE_CHARS:
        return "too long"
    if not ALLOWED_CHARACTERS.match(candidate):
        return "foreign script"
    if cell.intent == INTENT_KNOWLEDGE_QUESTION and TRANSLATION_CUES.search(candidate):
        return "reads as a translation request"
    if wrong_language(candidate, cell):
        return "wrong language"
    if guard.rejects(candidate):
        return "duplicate of known text"
    return None


def instruction_for(cell: Cell) -> str:
    """Return the generation instruction for one square of the grid."""
    if cell.intent == INTENT_KNOWLEDGE_QUESTION:
        return KNOWLEDGE_PROMPT
    if cell.intent == INTENT_OUT_OF_SCOPE:
        return OUT_OF_SCOPE_PROMPT
    return TRANSLATION_PROMPTS[cell.language]


def build_prompt(generator: Generator, cell: Cell, count: int) -> str:
    """Assemble one generation prompt for one square of the grid."""
    return (
        STYLE_PREAMBLE.format(examples="\n".join(generator.style_examples))
        + instruction_for(cell).format(n=count)
        + LANGUAGE_RULE.format(language=LANGUAGES[cell.language])
        + FORMAT_RULE
    )


def split_lines(response: str) -> list[str]:
    """Turn a model reply into candidate messages, one per line."""
    return [line.strip().strip('-").').strip() for line in response.splitlines()]


def generate_cell(generator: Generator, cell: Cell) -> tuple[list[str], dict]:
    """Fill one square of the grid.

    Args:
        generator: The model, the style sample and the duplicate screen.
        cell: The intent and language to generate for.

    Returns:
        The accepted messages and a count of rejections by reason. The list
        falls short of the cell's quota when the model kept failing the
        screens, which the caller reports rather than papers over.
    """
    accepted: dict[str, str] = {}
    rejected = dict.fromkeys(REJECTIONS, 0)
    rejected["calls"] = 0
    max_attempts = cell.quota // BATCH_SIZE + 8
    while len(accepted) < cell.quota and rejected["calls"] < max_attempts:
        rejected["calls"] += 1
        wanted = min(BATCH_SIZE, cell.quota - len(accepted))
        reply = generator.llm.invoke(
            [HumanMessage(content=build_prompt(generator, cell, wanted))]
        ).content
        for candidate in split_lines(reply):
            if not candidate or normalise(candidate) in accepted:
                continue
            reason = rejection_reason(candidate, cell, generator.guard)
            if reason:
                rejected[reason] += 1
                continue
            accepted[normalise(candidate)] = candidate
            generator.guard.add(candidate)
    return list(accepted.values())[: cell.quota], rejected


def report_cell(cell: Cell, written: int, rejected: dict) -> None:
    """Print what one square produced and what it threw away."""
    losses = ", ".join(
        f"{count} {reason}"
        for reason, count in rejected.items()
        if count and reason != "calls"
    )
    print(
        f"  {str(cell):<28} {written:>3} written, "
        f"{rejected['calls']} calls, rejected: {losses or 'none'}"
    )


def build_generator() -> Generator:
    """Read the corpora, draw the style sample, and assemble the screens."""
    if not REAL_PATH.exists():
        raise SystemExit(
            f"Real messages not found at {REAL_PATH}. They supply the wording "
            "this set is seeded from; set SETTLEIN_REAL_DATA to their location."
        )
    real = read_column(REAL_PATH, "text")
    training = read_column(TRAINING_PATH, "text")
    sample = random.Random(STYLE_SAMPLE_SEED).sample(
        real, min(STYLE_SAMPLE_SIZE, len(real))
    )
    print(
        f"seeded from {len(real)} real messages ({len(sample)} shown as style), "
        f"screened against {len(training)} training messages"
    )
    return Generator(
        llm=ChatOpenAI(
            model=GENERATOR_MODEL,
            temperature=GENERATOR_TEMPERATURE,
            openai_api_key=settings.openai_api_key,
        ),
        # One exemplar per line, so a message typed across several lines does
        # not read as several separate examples.
        style_examples=tuple(" ".join(message.split()) for message in sample),
        guard=DuplicateFilter.over([real, training]),
    )


def verify() -> None:
    """Check the written file against the screens that produced it.

    Regenerating is stochastic, so a re-run never reproduces the file. What
    can be held is that every row still passes the screens as they now stand.
    """
    rows = list(csv.DictReader(open(OUTPUT_PATH, encoding="utf-8", newline="")))
    print(f"{OUTPUT_PATH}: {len(rows)} rows")
    failures = 0

    written = collections.Counter((r["intent"], r["language"]) for r in rows)
    for intent, quotas in GRID.items():
        for language, quota in quotas.items():
            got = written[(intent, language)]
            if got != quota:
                print(f"  QUOTA  {intent}/{language}: {got} against {quota}")
                failures += 1
    for intent, language in written:
        if language not in GRID.get(intent, {}):
            print(f"  GRID   {intent}/{language} is not a square of the grid")
            failures += 1

    guard = DuplicateFilter.over(
        [read_column(REAL_PATH, "text"), read_column(TRAINING_PATH, "text")]
    )
    for row in rows:
        cell = Cell(row["intent"], row["language"])
        reason = rejection_reason(row["text"], cell, guard)
        if reason:
            print(f"  SCREEN {reason}: {row['text'][:60]}")
            failures += 1
        guard.add(row["text"])

    print(f"{failures} failure(s)" if failures else "every row passes every screen")
    if failures:
        raise SystemExit(1)


def main() -> None:
    """Fill every square of the grid, write the set, and report what it holds."""
    generator = build_generator()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    short = []
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "intent", "language"])
        for intent, quotas in GRID.items():
            for language, quota in quotas.items():
                cell = Cell(intent, language, quota)
                messages, rejected = generate_cell(generator, cell)
                for text in messages:
                    writer.writerow([text, cell.intent, cell.language])
                report_cell(cell, len(messages), rejected)
                if len(messages) < quota:
                    short.append(str(cell))

    print(f"\nSaved to {OUTPUT_PATH.resolve()}")
    print("Every label still has to be read by hand before anything is scored.")
    if short:
        print(f"SHORT of quota: {', '.join(short)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        main()
