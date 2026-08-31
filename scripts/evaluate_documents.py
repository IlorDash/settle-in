"""Read photographed documents through the bot's own graph - group E.

Section 4.5 asks three things of the document path that E0 and E1 cannot
answer, because both of those were taken with `scripts/probe_vision.py`, which
transcribes an image and stops there:

  * E2, what a user waits for after sending a photograph, mean and p95.
  * E3, what one photograph costs, with the image tokens counted apart from
    the text tokens - the image is what makes a reading expensive.
  * E4, what the bot replies to an image that carries no readable document.

So this script runs the graph instead. `process_document` is the call the
photo handler makes, on an orchestrator built the way the bot builds it, so a
figure here is a figure about the shipped system rather than about a harness.

One photograph is two model calls, and telling them apart is what E3 is:

  * the transcription, on the vision model, at the image's own resolution
  * the answer, on the cheaper text model, which never sees the image

Four things this script has to get right:

  * Nothing bought may be lost. Every finished photograph is appended to
    data/eval_documents_log.jsonl before anything is summarised, and
    `--from-log` rebuilds the report out of that file while sending nothing.
    A group A run once bought 90 messages and then threw them all away on an
    unpriced model name.
  * The two calls are priced apart. The callback reads the model name off
    each reply, so the vision call and the answer are never averaged into one
    price, and an unpriced model raises instead of counting as free.
  * The image tokens are derived, not reported. The API bills them as prompt
    tokens and breaks out none of them, so the transcription prompt is
    counted with tiktoken and subtracted. IMAGE_TOKEN_RULE states the rule
    and what it cannot distinguish; the journal entry quotes it.
  * It scores no accuracy. The photographs in thesis/assets/ have no expected
    values beside them, so nothing here counts a value as read. E1 is the
    accuracy measurement and it stands on its own three hand-checked bills.

Run from the repo root:

    python -m scripts.evaluate_documents --resolution-only    # free, extends E0
    python -m scripts.evaluate_documents --limit 2            # a smoke run
    python -m scripts.evaluate_documents                      # E2 and E3
    python -m scripts.evaluate_documents --set non-documents  # E4
    python -m scripts.evaluate_documents --set degraded       # E4
    python -m scripts.evaluate_documents --from-log --set all # free, rebuilds
    python -m scripts.evaluate_documents --tally              # free, reads marks

Set PYTHONIOENCODING=utf-8 first: the bills are in Serbian Cyrillic, and on a
Windows console the saved log is otherwise written in the local codepage.
"""

import argparse
import asyncio
import json
import math
import mimetypes
import platform
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler

from scripts.evaluate_latency import build_the_bot, percentile
from scripts.evaluate_system import PRICE_PER_MILLION
from scripts.probe_vision import effective_size, looks_degenerate
from src.agents.multimodal_agent import (
    DEFAULT_QUESTION,
    IMAGE_DETAIL,
    LLM_MODEL,
    SUMMARY_MODEL,
    TRANSCRIPTION_MAX_TOKENS,
    TRANSCRIPTION_PROMPT,
    encode_image_as_data_url,
)
from src.agents.orchestrator import DocumentTurn, process_document

SOURCE_DIR = Path("thesis/assets")
DEGRADED_DIR = SOURCE_DIR / "degraded"
REPORT_PATH = Path("data/eval_documents.json")
REVIEW_PATH = Path("data/eval_documents_review.md")
# Every finished photograph is written here the moment its two calls return.
# A run buys its readings and cannot get them back, so nothing downstream of
# the sending - summarising, pricing, printing, writing the sheet - may be
# able to discard them by failing. `--from-log` rebuilds the report from this
# file and sends nothing.
LOG_PATH = Path("data/eval_documents_log.jsonl")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DOCUMENT_PREFIXES = ("eps_", "infostan_")
NON_DOCUMENT_PREFIX = "not_a_document_"

SET_DOCUMENTS = "documents"
SET_NON_DOCUMENTS = "non-documents"
SET_DEGRADED = "degraded"
SETS = (SET_DOCUMENTS, SET_NON_DOCUMENTS, SET_DEGRADED)
# The two sets that carry no readable document, which is E4's question. The
# third set is E2 and E3, and no verdict is asked of it.
E4_SETS = (SET_NON_DOCUMENTS, SET_DEGRADED)

# Every photograph is sent with no caption, which is the case the bot answers
# with DEFAULT_QUESTION. It is the plainest document turn there is, it needs
# no question written per bill, and it is what a user who simply forwards a
# photo gets. A caption would make the reply a different measurement.
CAPTION = ""

# The encoding gpt-4o-class and later models use. tiktoken maps a model name
# to its encoding from a fixed table, which cannot know a model released
# after the version installed here, so the fallback is named rather than
# guessed and the run prints which of the two it used.
FALLBACK_ENCODING = "o200k_base"

# OpenAI's own recipe for counting a chat request: every message costs three
# tokens of framing, and three more prime the reply. The transcription
# request is two messages, the system prompt and the human turn carrying the
# image.
TOKENS_PER_MESSAGE = 3
TOKENS_FOR_THE_REPLY = 3
TRANSCRIPTION_MESSAGES = 2

# The vision model prices an image in patches of this size at detail=original.
PATCH_PIXELS = 32

IMAGE_TOKEN_RULE = (
    "The API bills image tokens as prompt tokens and breaks out none of "
    "them: input_token_details reports only audio and cache fields. So the "
    "image tokens are what is left after the text. The transcription prompt "
    "is a fixed string; its tokens are counted once with tiktoken, together "
    "with the three-tokens-per-message chat framing, and that count is "
    "subtracted from the input_tokens the reply reports. "
    "What the rule cannot distinguish: any per-request overhead the API adds "
    "beyond that framing is counted as image, as is the data URL's own "
    "header, and a prompt-caching discount - were one ever applied to this "
    "call - would come off the image share rather than off the text. The "
    "text baseline is a constant, so an error in it is the same constant on "
    "every document and does not affect how the image tokens vary with the "
    "size of the photograph."
)


def _natural_key(path: Path) -> tuple:
    """Sort eps_2 before eps_10, so `--limit 2` takes the first two bills."""
    digits = [int(number) for number in re.findall(r"\d+", path.stem)]
    return re.sub(r"\d+", "", path.stem), digits


def images_in(directory: Path, keep) -> list[Path]:
    """List the images of one set, in a fixed order.

    Args:
        directory: Where to look. Never recursive: the degraded set lives in
            a folder inside the source folder, and the two are separate sets.
        keep: Predicate on the file stem.

    Returns:
        The matching images, naturally sorted.
    """
    if not directory.exists():
        return []
    found = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and keep(path.stem)
    ]
    return sorted(found, key=_natural_key)


def load_set(name: str) -> list[Path]:
    """Return the images of one named set.

    Args:
        name: One of SETS.

    Returns:
        The images belonging to it.
    """
    if name == SET_DOCUMENTS:
        return images_in(SOURCE_DIR, lambda stem: stem.startswith(DOCUMENT_PREFIXES))
    if name == SET_NON_DOCUMENTS:
        return images_in(SOURCE_DIR, lambda stem: stem.startswith(NON_DOCUMENT_PREFIX))
    return images_in(DEGRADED_DIR, lambda stem: True)


def price_for(model: str) -> dict:
    """The published price of one model, snapshot names included.

    OpenAI answers a request for "gpt-4o-mini" with a reply naming the
    snapshot behind it, and the price table is keyed by the alias, so the
    longest alias the name begins with is the model that was billed - longest
    and not first, because "gpt-4o" is itself a prefix of "gpt-4o-mini".

    Args:
        model: The model name as the reply gave it.

    Returns:
        Its input and output price per million tokens.

    Raises:
        KeyError: If no priced model matches. Deliberately loud, and it names
            this script's own log: an unpriced model counted as free would
            understate every document silently, and the readings already paid
            for are recoverable from LOG_PATH.
    """
    if model in PRICE_PER_MILLION:
        return PRICE_PER_MILLION[model]
    matches = [name for name in PRICE_PER_MILLION if model.startswith(name)]
    if not matches:
        raise KeyError(
            f"no published price for {model!r}. Add it to PRICE_PER_MILLION "
            f"in scripts/evaluate_system.py; the readings of this run are in "
            f"{LOG_PATH} and --from-log will rebuild the report for free."
        )
    return PRICE_PER_MILLION[max(matches, key=len)]


def prompt_text_tokens(model: str) -> tuple[int, str]:
    """Count the transcription request with the image taken out of it.

    This is the baseline IMAGE_TOKEN_RULE subtracts. It is a constant: the
    transcription prompt never varies, and the human turn carries the image
    and nothing else.

    Args:
        model: The vision model, so the right encoding is used where tiktoken
            knows it.

    Returns:
        The token count, and the name of the encoding it was counted with.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        used = encoding.name
    except KeyError:
        encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
        used = FALLBACK_ENCODING
    framing = TOKENS_PER_MESSAGE * TRANSCRIPTION_MESSAGES + TOKENS_FOR_THE_REPLY
    return len(encoding.encode(TRANSCRIPTION_PROMPT)) + framing, used


def patch_count(width: int, height: int) -> int:
    """Patches a page of this size divides into, before any budget applies.

    A free cross-check on the subtraction rule, not a second measurement. The
    model shrinks an image that would exceed its patch budget, so this number
    and the measured image tokens are expected to part company above some
    size - and where they part is where that budget is, which is worth
    reporting.

    Args:
        width: The image's real width in pixels.
        height: The image's real height in pixels.

    Returns:
        ceil(width / 32) * ceil(height / 32).
    """
    return math.ceil(width / PATCH_PIXELS) * math.ceil(height / PATCH_PIXELS)


@dataclass
class CallRecorder(BaseCallbackHandler):
    """Records every model call of one photograph, in the order they finished.

    Both chains of the document agent end in StrOutputParser(), which turns
    the AIMessage into text and drops usage_metadata with it. A handler gets
    there first. It is also the only way to see the transcript without
    touching src/: process_document() returns the answer alone, and the
    transcript stays inside the graph.

    Attributes:
        calls: One entry per model call - the model the reply names, the
            tokens billed, the wall clock, and what came back.
    """

    calls: list = field(default_factory=list)
    _started: list = field(default_factory=list, init=False)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        """Mark the start of a call. Both chains use the chat interface."""
        self._started.append(time.perf_counter())

    def on_llm_end(self, response, **kwargs) -> None:
        """Close the call and read the model, the usage and the reply off it."""
        started = self._started.pop() if self._started else time.perf_counter()
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        self.calls.append(
            {
                "model": (response.llm_output or {}).get("model_name") or "",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "text": generation.text,
            }
        )


# What the answer prompt tells the model to do with an image that carries
# nothing: "If the transcript shows no document or product at all, say so in
# one sentence." Every photograph is sent without a caption, so the question
# is DEFAULT_QUESTION, which is English, and the reply comes back in English.
# This is a first cut only. Every reply is written into the review sheet and
# the hand pass there is what E4 counts, for the reason group C stopped
# trusting its own decline markers: a wording nobody anticipated reads as its
# opposite.
NO_DOCUMENT_MARKERS = (
    "no document",
    "not a document",
    "isn't a document",
    "is not a document",
    "does not show a document",
    "doesn't show a document",
    "no readable",
    "not readable",
    "unreadable",
    "illegible",
    "cannot be read",
    "can't be read",
    "could not be read",
    "no text",
    "not legible",
)

# The transcription prompt asks for [?] wherever a character could not be
# read, so a count of them says how much of the page the model admitted it
# could not see. An honest failure is full of them; an invented one has none.
UNREADABLE_MARK = "[?]"


@dataclass(frozen=True)
class Reading:
    """One photograph, read through the graph.

    Attributes:
        name: The file, as the report and the review sheet name it.
        image_set: Which of SETS it belongs to.
        width: The file's own width in pixels, or 0 without Pillow.
        height: The file's own height in pixels, or 0 without Pillow.
        file_bytes: Its size on disk.
        encode_ms: Reading the file and base64-encoding it. Outside total_ms,
            because in the deployed bot this sits with the Telegram download
            and neither is part of the graph.
        total_ms: The graph, photograph in and reply out. This is E2.
        transcription: The verbatim reading, caught off the vision call.
        answer: The reply the user would receive.
        calls: One dict per model call, from CallRecorder.
        prompt_text_tokens: The baseline IMAGE_TOKEN_RULE subtracts.
        degenerate: Whether the transcript collapsed into a repeated line.
        unreadable_marks: How many [?] the transcript carries.
        reads_as_no_document: Whether the answer matches a NO_DOCUMENT_MARKER.
    """

    name: str
    image_set: str
    width: int
    height: int
    file_bytes: int
    encode_ms: float
    total_ms: float
    transcription: str
    answer: str
    calls: list
    prompt_text_tokens: int
    degenerate: bool
    unreadable_marks: int
    reads_as_no_document: bool

    @property
    def vision_call(self) -> dict:
        """The call that saw the image.

        The first one: handle_document awaits the transcription before it
        starts the answer, so this is the node's own order and not a guess.
        `unexpected_pairings` in the report names any reading whose calls did
        not land on the two configured models.
        """
        return self.calls[0]

    @property
    def summary_call(self) -> dict:
        """The call that wrote the answer from the transcript."""
        return self.calls[-1]

    @property
    def transcription_ms(self) -> float:
        """Time inside the vision call."""
        return self.vision_call["elapsed_ms"]

    @property
    def summary_ms(self) -> float:
        """Time inside the answering call."""
        return self.summary_call["elapsed_ms"]

    @property
    def other_ms(self) -> float:
        """What the two model calls leave over.

        LangGraph's dispatch into and out of the node, the state merge and
        the checkpointer write. Reported as its own line rather than
        distributed over the two calls, which would be a guess.
        """
        return self.total_ms - self.transcription_ms - self.summary_ms

    @property
    def image_tokens(self) -> int:
        """Prompt tokens the photograph itself cost. See IMAGE_TOKEN_RULE."""
        return self.vision_call["input_tokens"] - self.prompt_text_tokens

    @property
    def megapixels(self) -> float:
        """The photograph's own size, which is what the image tokens track."""
        return self.width * self.height / 1e6

    @property
    def usd(self) -> float:
        """What this photograph cost, each call at its own model's price."""
        total = 0.0
        for call in self.calls:
            price = price_for(call["model"])
            total += (
                call["input_tokens"] * price["input"]
                + call["output_tokens"] * price["output"]
            ) / 1_000_000
        return total


def record(reading: Reading) -> None:
    """Append one finished photograph to the file that survives a failure."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(reading), ensure_ascii=False) + "\n")


def read_log(path: Path, wanted: tuple) -> list[Reading]:
    """Rebuild the readings from the log; the latest reading of each wins.

    Not "the last run", because the sets are bought in separate runs - the
    documents in one, E4's images in another - and a rebuild has to be able
    to hold all of them at once. Keying on the set and the file means a
    photograph read twice is counted once, at its most recent reading.

    Args:
        path: The JSONL the runs wrote.
        wanted: The sets to return.

    Returns:
        The readings, set by set and naturally sorted within each.
    """
    latest: dict = {}
    superseded = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stored = json.loads(line)
            key = (stored["image_set"], stored["name"])
            superseded += key in latest
            latest[key] = stored
    if superseded:
        print(f"  {superseded} earlier reading(s) superseded by a later one")
    chosen = [row for (image_set, _), row in latest.items() if image_set in wanted]
    chosen.sort(
        key=lambda row: (SETS.index(row["image_set"]), _natural_key(Path(row["name"])))
    )
    return [Reading(**row) for row in chosen]


def image_size(path: Path) -> tuple:
    """The photograph's own pixel size, or (0, 0) without Pillow.

    Pillow is not a declared dependency of this project, and losing the
    resolution columns is a smaller loss than adding one, so a missing Pillow
    degrades the report rather than stopping a paid run.

    Args:
        path: The image file.

    Returns:
        Its (width, height) in pixels.
    """
    try:
        from PIL import Image
    except ImportError:
        return 0, 0
    with Image.open(path) as image:
        return image.size


async def read_one(orchestrator, path: Path, image_set: str, baseline: int) -> Reading:
    """Send one photograph through the graph and measure everything about it.

    Args:
        orchestrator: The compiled graph the bot runs.
        path: The image file.
        image_set: Which of SETS it came from.
        baseline: The prompt-text token count IMAGE_TOKEN_RULE subtracts.

    Returns:
        One reading.
    """
    started = time.perf_counter()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_bytes = path.read_bytes()
    image_url = encode_image_as_data_url(image_bytes, mime_type)
    encode_ms = (time.perf_counter() - started) * 1000

    # The handler is bound to the graph rather than passed at the call, so
    # the bot's own process_document() is still what runs the photograph.
    # LangGraph carries callbacks down into every node.
    recorder = CallRecorder()
    watched = orchestrator.with_config(callbacks=[recorder])
    started = time.perf_counter()
    result = await process_document(
        watched,
        DocumentTurn(image_url=image_url, caption=CAPTION),
        thread_id=f"documents-{image_set}-{path.stem}",
    )
    total_ms = (time.perf_counter() - started) * 1000

    width, height = image_size(path)
    transcription = recorder.calls[0]["text"] if recorder.calls else ""
    lowered = result.response.lower()
    return Reading(
        name=path.name,
        image_set=image_set,
        width=width,
        height=height,
        file_bytes=len(image_bytes),
        encode_ms=encode_ms,
        total_ms=total_ms,
        transcription=transcription,
        answer=result.response,
        calls=[
            {key: value for key, value in call.items() if key != "text"}
            for call in recorder.calls
        ],
        prompt_text_tokens=baseline,
        degenerate=looks_degenerate(transcription),
        unreadable_marks=transcription.count(UNREADABLE_MARK),
        reads_as_no_document=any(marker in lowered for marker in NO_DOCUMENT_MARKERS),
    )


async def read_all(paths: list, image_set: str, baseline: int) -> list[Reading]:
    """Read every photograph of one set, one at a time.

    One at a time, because E2 is what a single user waits for. Sending them
    concurrently would measure queuing, which is group A's question.

    Args:
        paths: The images to read.
        image_set: Which of SETS they came from.
        baseline: The prompt-text token count IMAGE_TOKEN_RULE subtracts.

    Returns:
        One reading per photograph.
    """
    orchestrator = build_the_bot()
    readings = []
    for position, path in enumerate(paths, start=1):
        reading = await read_one(orchestrator, path, image_set, baseline)
        record(reading)
        readings.append(reading)
        print(
            f"  {position}/{len(paths)} {path.name}"
            f"  {reading.total_ms / 1000:.1f}s"
            f"  {reading.image_tokens} image tokens"
            f"  ${reading.usd:.4f}"
        )
    return readings


def resolution_rows(paths: list) -> list:
    """What a gpt-4o-class model would be given for each photograph.

    This is E0's arithmetic and it costs nothing. It describes the model the
    deployed setting replaced: detail="original" skips the reduction, so
    these figures are what the cheaper generation loses, which is the loss E1
    scored.

    Args:
        paths: The images to measure.

    Returns:
        One row per image, or an empty list without Pillow.
    """
    rows = []
    for path in paths:
        width, height = image_size(path)
        if not width:
            continue
        seen_width, seen_height = effective_size(width, height)
        rows.append(
            {
                "file": path.name,
                "width": width,
                "height": height,
                "megapixels": round(width * height / 1e6, 2),
                "gpt4o_width": seen_width,
                "gpt4o_height": seen_height,
                "pixels_kept": (seen_width * seen_height) / (width * height),
                "patches_at_own_resolution": patch_count(width, height),
            }
        )
    return rows


def summarise_resolution(rows: list) -> dict:
    """Reduce the resolution table to the figures E0 quotes."""
    kept = [row["pixels_kept"] for row in rows]
    return {
        "n": len(rows),
        "note": (
            "what a gpt-4o-class model is given after the API fits the image "
            "in a 2048px square and reduces it to a 768px shortest side. The "
            "deployed setting, detail=original on the newer model, skips this "
            "reduction; these figures are what the cheaper generation loses."
        ),
        "pixels_kept_mean": statistics.fmean(kept),
        "pixels_kept_min": min(kept),
        "pixels_kept_max": max(kept),
        "at_or_below_10_percent": sum(1 for share in kept if share <= 0.10),
        "megapixels_mean": statistics.fmean([row["megapixels"] for row in rows]),
        "rows": rows,
    }


def cost_by_model(readings: list) -> dict:
    """Price each model's share of the set separately.

    The whole of E3: one photograph is a vision call and a text call, at
    prices six times apart on output, and one figure over both would say
    nothing about either.

    Args:
        readings: The set's readings.

    Returns:
        One block per model that billed anything.
    """
    by_model: dict = {}
    for reading in readings:
        for call in reading.calls:
            block = by_model.setdefault(
                call["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            block["calls"] += 1
            block["input_tokens"] += call["input_tokens"]
            block["output_tokens"] += call["output_tokens"]
    for model, block in by_model.items():
        price = price_for(model)
        block["input_tokens_mean"] = block["input_tokens"] / len(readings)
        block["output_tokens_mean"] = block["output_tokens"] / len(readings)
        block["usd"] = (
            block["input_tokens"] * price["input"]
            + block["output_tokens"] * price["output"]
        ) / 1_000_000
        block["usd_per_document"] = block["usd"] / len(readings)
    return by_model


def summarise_image_tokens(readings: list) -> dict:
    """Reduce the image-token counts to the figures E3 quotes."""
    image = [reading.image_tokens for reading in readings]
    vision_input = [reading.vision_call["input_tokens"] for reading in readings]
    sized = [reading for reading in readings if reading.width]
    return {
        "rule": IMAGE_TOKEN_RULE,
        "prompt_text_tokens": readings[0].prompt_text_tokens,
        "image_tokens_mean": statistics.fmean(image),
        "image_tokens_min": min(image),
        "image_tokens_max": max(image),
        "share_of_vision_input": sum(image) / sum(vision_input),
        # The cross-check patch_count() describes. Where the two part company
        # is where the model's own patch budget is.
        "patch_prediction": [
            {
                "file": reading.name,
                "megapixels": round(reading.megapixels, 2),
                "patches": patch_count(reading.width, reading.height),
                "image_tokens": reading.image_tokens,
            }
            for reading in sized
        ],
    }


def summarise_set(readings: list) -> dict:
    """Reduce one set's readings to the figures E2, E3 and E4 report."""
    totals = [reading.total_ms for reading in readings]
    block = {
        "n": len(readings),
        "latency": {
            "mean_ms": statistics.fmean(totals),
            "median_ms": percentile(totals, 0.5),
            "p95_ms": percentile(totals, 0.95),
            "min_ms": min(totals),
            "max_ms": max(totals),
        },
        "stages": {
            "transcription_ms": statistics.fmean(
                [reading.transcription_ms for reading in readings]
            ),
            "summary_ms": statistics.fmean(
                [reading.summary_ms for reading in readings]
            ),
            "other_ms": statistics.fmean([reading.other_ms for reading in readings]),
            "encode_ms": statistics.fmean([reading.encode_ms for reading in readings]),
            "note": (
                "the two model calls are timed by a callback; other_ms is "
                "what they leave over inside the graph. encode_ms is reading "
                "the file and base64-encoding it, which sits outside the "
                "graph and outside mean_ms, as the Telegram download does."
            ),
        },
        "cost": {
            "by_model": cost_by_model(readings),
            "usd_per_document_mean": statistics.fmean(
                [reading.usd for reading in readings]
            ),
            "usd_per_document_min": min(reading.usd for reading in readings),
            "usd_per_document_max": max(reading.usd for reading in readings),
            "usd_for_this_set": sum(reading.usd for reading in readings),
        },
        "image_tokens": summarise_image_tokens(readings),
        "transcripts_that_collapsed": [
            reading.name for reading in readings if reading.degenerate
        ],
        "unreadable_marks_mean": statistics.fmean(
            [reading.unreadable_marks for reading in readings]
        ),
        "unexpected_pairings": [
            {"file": reading.name, "models": [c["model"] for c in reading.calls]}
            for reading in readings
            if len(reading.calls) != 2
            or not reading.calls[0]["model"].startswith(LLM_MODEL)
            or not reading.calls[-1]["model"].startswith(SUMMARY_MODEL)
        ],
    }
    if readings[0].image_set in E4_SETS:
        block["e4_first_cut"] = {
            "note": (
                "an automatic first cut only, from NO_DOCUMENT_MARKERS. The "
                "measurement is the hand pass in the review sheet, which "
                "--tally counts; this line exists so that a disagreement "
                "between the two is visible rather than buried."
            ),
            "reads_as_no_document": sum(
                1 for reading in readings if reading.reads_as_no_document
            ),
            "carries_no_unreadable_mark": sum(
                1 for reading in readings if not reading.unreadable_marks
            ),
        }
    return block


def build_report(readings: list, resolution: list) -> dict:
    """Assemble the report the journal entry is written from.

    Args:
        readings: Every reading of this run, or of the log.
        resolution: E0's free arithmetic over the same sets.

    Returns:
        The report saved to REPORT_PATH.
    """
    by_set: dict = {}
    for reading in readings:
        by_set.setdefault(reading.image_set, []).append(reading)
    return {
        "machine": {
            "label": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.machine(),
            "python": platform.python_version(),
            "note": (
                "group E depends on the model and the image, not on the "
                "machine, so unlike groups A and B it need not be taken on "
                "the deployment hardware. Recorded only so that a latency "
                "figure here is never mistaken for one of group A's."
            ),
        },
        "settings": {
            "vision_model": LLM_MODEL,
            "image_detail": IMAGE_DETAIL,
            "transcription_max_tokens": TRANSCRIPTION_MAX_TOKENS,
            "summary_model": SUMMARY_MODEL,
            "caption": CAPTION,
            "question_asked": DEFAULT_QUESTION,
            "note": (
                "every photograph is a first message: a fresh thread_id per "
                "image, so none of them carries a conversation behind it"
            ),
        },
        "sets": {name: summarise_set(rows) for name, rows in by_set.items()},
        "resolution": summarise_resolution(resolution) if resolution else {},
        "per_image": [
            {
                "file": reading.name,
                "set": reading.image_set,
                "megapixels": round(reading.megapixels, 2),
                "total_ms": reading.total_ms,
                "transcription_ms": reading.transcription_ms,
                "summary_ms": reading.summary_ms,
                "other_ms": reading.other_ms,
                "vision_input_tokens": reading.vision_call["input_tokens"],
                "image_tokens": reading.image_tokens,
                "vision_output_tokens": reading.vision_call["output_tokens"],
                "summary_input_tokens": reading.summary_call["input_tokens"],
                "summary_output_tokens": reading.summary_call["output_tokens"],
                "usd": reading.usd,
                "unreadable_marks": reading.unreadable_marks,
                "degenerate": reading.degenerate,
                "reads_as_no_document": reading.reads_as_no_document,
            }
            for reading in readings
        ],
        "cost_of_this_run_usd": sum(reading.usd for reading in readings),
    }


VERDICTS = ("refused", "described", "invented")
SECTION = re.compile(r"^## \[([\w-]+)\] (\S+)$", re.M)


def write_review_sheet(readings: list, path: Path) -> None:
    """Write the sheet the author reads, and E4's hand pass is made on.

    Every reply is printed beside the file it came from and the transcript it
    was written from. The document sections carry no verdict box: those
    photographs have no expected values, so there is nothing to score there,
    and E1 is where reading accuracy lives.

    Args:
        readings: Every reading to print.
        path: Where to write the sheet.
    """
    lines = [
        "# Group E, the review sheet",
        "",
        "One section per photograph: the reply the bot would have sent, and",
        "the transcript it was written from. Written by",
        "`scripts/evaluate_documents.py`.",
        "",
        f"**The `{SET_DOCUMENTS}` sections are for reading, not for scoring.**",
        "Those bills have no expected values beside them, so nothing counts a",
        "value as read here. E1 is the accuracy measurement and it stands on",
        "its own three hand-checked bills.",
        "",
        "**The other sections are E4, and they are marked by hand.** Put an",
        "`x` in exactly one box:",
        "",
        "- `refused` - the reply says there is no document, or that the image",
        "  could not be read. The wanted behaviour.",
        "- `described` - the reply describes what is actually in the picture",
        "  without claiming it is a document. Also honest.",
        "- `invented` - the reply states figures, dates, names or fields as if",
        "  it had read a document. The failure E4 is looking for. A reply that",
        "  hedges and then quotes a total anyway is `invented`.",
        "",
        "Add a line under **Notes** where the verdict needed a decision. Then",
        "run:",
        "",
        "```",
        ".venv/Scripts/python.exe -m scripts.evaluate_documents --tally",
        "```",
        "",
        "which counts the marks and refuses a section that is unmarked or",
        "marked twice.",
        "",
        "---",
        "",
    ]
    for reading in readings:
        lines += [
            f"## [{reading.image_set}] {reading.name}",
            "",
            f"- **Size:** {reading.width}x{reading.height}"
            f" ({reading.megapixels:.1f} MP), {reading.file_bytes / 1024:.0f} KB",
            f"- **Waited:** {reading.total_ms / 1000:.1f} s"
            f" (transcription {reading.transcription_ms / 1000:.1f} s,"
            f" answer {reading.summary_ms / 1000:.1f} s)",
            f"- **Tokens:** {reading.image_tokens} image,"
            f" {reading.vision_call['output_tokens']} transcript out,"
            f" {reading.summary_call['output_tokens']} answer out",
            f"- **Cost:** ${reading.usd:.4f}",
            f"- **`[?]` in the transcript:** {reading.unreadable_marks}",
            f"- **Transcript collapsed:** {'yes' if reading.degenerate else 'no'}",
            "- **Reads as 'no document':** "
            f"{'yes' if reading.reads_as_no_document else 'no'}",
            "",
        ]
        if reading.image_set in E4_SETS:
            lines += [
                "**Verdict:** [ ] refused  [ ] described  [ ] invented",
                "",
                "**Notes:**",
                "",
            ]
        lines += [
            "### The reply the user would get",
            "",
            reading.answer.strip() or "_empty_",
            "",
            "### The transcript it was written from",
            "",
            "```",
            reading.transcription.strip() or "(empty)",
            "```",
            "",
            "---",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    # A marked sheet is hand work that no re-run may destroy. If any box has
    # been ticked, the new sheet goes beside it and the reader decides.
    if path.exists() and re.search(r"\[[xX]\]", path.read_text(encoding="utf-8")):
        beside = path.with_suffix(".new.md")
        beside.write_text("\n".join(lines), encoding="utf-8")
        print(
            f"\n{path} already carries verdicts, so it was left alone.\n"
            f"The sheet for this run is {beside}."
        )
        return
    path.write_text("\n".join(lines), encoding="utf-8")


def tally(path: Path) -> None:
    """Count the E4 verdicts marked by hand in the review sheet.

    Args:
        path: The sheet the pass was made on.

    Raises:
        SystemExit: If the sheet is missing, or a section is unmarked or
            marked twice - a half-marked sheet must not produce a figure.
    """
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Run a measurement first.")
    text = path.read_text(encoding="utf-8")
    starts = [(m.group(1), m.group(2), m.start()) for m in SECTION.finditer(text)]

    counts = {name: dict.fromkeys(VERDICTS, 0) for name in E4_SETS}
    unmarked, doubled, notes = [], [], {}
    for position, (image_set, name, start) in enumerate(starts):
        if image_set not in E4_SETS:
            continue
        end = starts[position + 1][2] if position + 1 < len(starts) else len(text)
        block = text[start:end]
        marked = [v for v in VERDICTS if re.search(rf"\[[xX]\]\s*{v}\b", block)]
        if not marked:
            unmarked.append(name)
        elif len(marked) > 1:
            doubled.append(name)
        else:
            counts[image_set][marked[0]] += 1
        found = re.search(r"\*\*Notes:\*\*(.*?)(?=\n#{2,3} |\n---|\Z)", block, re.S)
        if found and found.group(1).strip():
            notes[name] = found.group(1).strip()

    print(f"E4 verdicts in {path}")
    if unmarked:
        print(f"  UNMARKED: {', '.join(unmarked)}")
    if doubled:
        print(f"  MARKED TWICE: {', '.join(doubled)}")
    for image_set in E4_SETS:
        decided = sum(counts[image_set].values())
        if not decided:
            continue
        readable = "  ".join(f"{v} {counts[image_set][v]}" for v in VERDICTS)
        print(f"\n[{image_set}] {decided} marked: {readable}")
        print(
            f"  invented {counts[image_set]['invented']} of {decided}"
            f" ({counts[image_set]['invented'] / decided:.4f})"
        )
    if notes:
        print(f"\n{len(notes)} section(s) carry notes:")
        for name, note in notes.items():
            print(f"  {name}: {note.splitlines()[0]}")
    if unmarked or doubled:
        raise SystemExit(1)


def print_resolution(report: dict) -> None:
    """Print E0's arithmetic, which costs nothing and needs no run."""
    resolution = report.get("resolution")
    if not resolution:
        return
    print(f"\n[E0] {resolution['n']} photographs, on a gpt-4o-class model")
    print(
        f"  pixels kept: mean {resolution['pixels_kept_mean']:.1%},"
        f" from {resolution['pixels_kept_min']:.1%}"
        f" to {resolution['pixels_kept_max']:.1%}"
    )
    print(
        f"  {resolution['at_or_below_10_percent']} of {resolution['n']}"
        f" keep 10% of their pixels or less"
        f" (mean {resolution['megapixels_mean']:.1f} MP on file)"
    )


def print_report(report: dict, full_set_sizes: dict) -> None:
    """Print the figures the journal entry records.

    Args:
        report: What build_report() produced.
        full_set_sizes: How many images each set holds in full, so that a
            smoke run can say what the whole of it would cost.
    """
    print("\n" + "=" * 70)
    print("document path, group E")
    print("=" * 70)
    print_resolution(report)

    for name, block in report["sets"].items():
        latency, stages, cost = block["latency"], block["stages"], block["cost"]
        tokens = block["image_tokens"]
        print(f"\n[{name}] n={block['n']}")
        print(
            f"  E2  total    mean {latency['mean_ms'] / 1000:>6.1f} s"
            f"   median {latency['median_ms'] / 1000:>6.1f}"
            f"   p95 {latency['p95_ms'] / 1000:>6.1f}"
            f"   max {latency['max_ms'] / 1000:>6.1f}"
        )
        print(
            f"      stages   transcription {stages['transcription_ms'] / 1000:>5.1f} s"
            f"   answer {stages['summary_ms'] / 1000:>5.1f}"
            f"   other {stages['other_ms'] / 1000:>5.1f}"
            f"   (encode {stages['encode_ms']:.0f} ms, outside the graph)"
        )
        print(
            f"  E3  ${cost['usd_per_document_mean']:.4f} per document"
            f"   (from ${cost['usd_per_document_min']:.4f}"
            f" to ${cost['usd_per_document_max']:.4f})"
        )
        for model, priced in cost["by_model"].items():
            print(
                f"      {model:<26} in {priced['input_tokens_mean']:>8.0f}"
                f"   out {priced['output_tokens_mean']:>7.0f}"
                f"   ${priced['usd_per_document']:.4f}/doc"
            )
        print(
            f"      image tokens  mean {tokens['image_tokens_mean']:>8.0f}"
            f"   min {tokens['image_tokens_min']:>6}"
            f"   max {tokens['image_tokens_max']:>6}"
            f"   ({tokens['share_of_vision_input']:.1%} of the vision prompt,"
            f" text baseline {tokens['prompt_text_tokens']})"
        )
        print(f"      [?] marks per transcript: {block['unreadable_marks_mean']:.1f}")
        for collapsed in block["transcripts_that_collapsed"]:
            print(f"      COLLAPSED TRANSCRIPT: {collapsed}")
        for odd in block["unexpected_pairings"]:
            print(f"      UNEXPECTED MODELS on {odd['file']}: {odd['models']}")
        if "e4_first_cut" in block:
            cut = block["e4_first_cut"]
            print(
                f"  E4  first cut: {cut['reads_as_no_document']} of {block['n']}"
                f" read as 'no document',"
                f" {cut['carries_no_unreadable_mark']} carry no [?] at all"
            )
            print("      the measurement is the hand pass; run --tally after it")

        remaining = full_set_sizes.get(name, 0) - block["n"]
        if remaining > 0:
            print(
                f"  ESTIMATE  the {remaining} image(s) not read would cost about"
                f" ${cost['usd_per_document_mean'] * remaining:.2f}"
                f" at this run's mean"
            )

    print(f"\n[cost] ${report['cost_of_this_run_usd']:.4f} for this run")


def parse_args() -> argparse.Namespace:
    """Read which images to measure, and which of the free modes to run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="image_set",
        default=SET_DOCUMENTS,
        choices=(*SETS, "all"),
        help=f"which images to read (default {SET_DOCUMENTS}, which is E2 and E3)",
    )
    parser.add_argument(
        "--limit", type=int, help="read only the first N images of each set"
    )
    parser.add_argument(
        "--resolution-only",
        action="store_true",
        help="print E0's arithmetic over the set and stop, sending nothing",
    )
    parser.add_argument(
        "--from-log",
        action="store_true",
        help=f"rebuild the report from {LOG_PATH}, sending nothing",
    )
    parser.add_argument(
        "--tally",
        action="store_true",
        help=f"count the E4 verdicts marked by hand in {REVIEW_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Measure the document path and save the report and the review sheet."""
    arguments = parse_args()
    if arguments.tally:
        tally(REVIEW_PATH)
        return

    wanted = SETS if arguments.image_set == "all" else (arguments.image_set,)
    full_set_sizes = {name: len(load_set(name)) for name in wanted}
    # E0's arithmetic is free, so it is taken over every photograph of the
    # chosen sets whatever else the run does, and not only over what was read.
    resolution = resolution_rows([path for name in wanted for path in load_set(name)])

    if arguments.resolution_only:
        if not resolution:
            raise SystemExit("Pillow is not installed, so no image size can be read")
        summary = summarise_resolution(resolution)
        print_resolution({"resolution": summary})
        for row in summary["rows"]:
            print(
                f"  {row['file']:<30} {row['width']:>5}x{row['height']:<5}"
                f" ({row['megapixels']:>5.2f} MP)"
                f" -> {row['gpt4o_width']:>4}x{row['gpt4o_height']:<4}"
                f"  {row['pixels_kept']:>4.0%} kept"
                f"  {row['patches_at_own_resolution']:>6} patches at own size"
            )
        return

    if arguments.from_log:
        if not LOG_PATH.exists():
            raise SystemExit(f"{LOG_PATH} does not exist, so there is nothing to read")
        readings = read_log(LOG_PATH, wanted)
        print(f"{len(readings)} reading(s) read from {LOG_PATH}, nothing sent")
    else:
        baseline, encoding_name = prompt_text_tokens(LLM_MODEL)
        print(
            f"transcription prompt is {baseline} tokens on {encoding_name};"
            f" everything above that in the vision call is the image"
        )
        readings = []
        for name in wanted:
            paths = load_set(name)
            if arguments.limit:
                paths = paths[: arguments.limit]
            if not paths:
                print(f"[{name}] no images found, skipped")
                continue
            print(f"\n[{name}] reading {len(paths)} image(s), one at a time")
            readings += asyncio.run(read_all(paths, name, baseline))

    if not readings:
        raise SystemExit("nothing was read, so there is no report to write")

    report = build_report(readings, resolution)
    print_report(report, full_set_sizes)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {REPORT_PATH.resolve()}")
    write_review_sheet(readings, REVIEW_PATH)
    print(f"The E4 pass is done in {REVIEW_PATH.resolve()}")
    if not arguments.from_log:
        print(f"Every reading is also in {LOG_PATH}, which --from-log rebuilds from")


if __name__ == "__main__":
    main()
