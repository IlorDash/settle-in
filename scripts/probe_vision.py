"""Measure how much of an image the vision model can actually read.

The document agent reads and reasons in one pass, so when its answer is wrong
there is nothing to inspect: a misread digit and a bad inference look
identical from the outside. This harness makes the reading step observable. It
asks the model to transcribe the image verbatim, once per `detail` setting,
then reports which expected values failed to appear - so the comparison is a
count, not an impression.

Run from the repo root:

    python -m scripts.probe_vision photo.jpg
    python -m scripts.probe_vision photo.jpg --expect expected.txt
    python -m scripts.probe_vision photo.jpg --detail low,high --model gpt-4o

The `--expect` file holds one value per line as printed on the document;
blank lines and lines starting with `#` are ignored.

Each detail setting costs one real API call.
"""

import argparse
import asyncio
import mimetypes
import re
from collections import Counter
from pathlib import Path

from src.agents.multimodal_agent import (
    LLM_MODEL,
    build_transcription_chain,
    encode_image_as_data_url,
    transcribe,
)

DEFAULT_DETAILS = "high"
# GPT-4o-class models fit the image inside this square, then shrink it until
# the shortest side hits SHORTEST_SIDE. Everything smaller than a pixel after
# that is simply gone, which is what makes a dense table unreadable.
MAX_SQUARE = 2048
SHORTEST_SIDE = 768


def load_expected(path: Path) -> list:
    """Read expected values, one per line, skipping blanks and comments.

    Args:
        path: File of values as printed on the document.

    Returns:
        The values, in file order.
    """
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)
    return values


# A minus sign is meaning, not decoration: on a bill it separates a discount
# from a charge. So the sign is compared, and only the glyph is forgiven -
# models write U+2212 and en dashes where the document prints a hyphen.
_MINUS_GLYPHS = str.maketrans({"−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-"})
_SPACED_MINUS = re.compile(r"-\s+(?=\d)")


def normalise(text: str) -> str:
    """Collapse whitespace and unify minus signs before comparing.

    A line break inside a wrapped table row, a Unicode minus, or a space
    between the sign and its digits are all transcription style rather than
    misreadings, so none of them should count as a miss.

    Args:
        text: A transcript, or one expected value.

    Returns:
        The text with whitespace collapsed and every dash-like character
        rewritten as a plain hyphen.
    """
    unified = " ".join(text.translate(_MINUS_GLYPHS).split())
    return _SPACED_MINUS.sub("-", unified)


def separator_variants(value: str) -> set:
    """Return the value written with either decimal separator.

    A minus sign carries meaning and is compared strictly, but "+12.58" and
    "+12,58" are the same number written two ways, and models routinely
    rewrite a document's decimal point as the comma used elsewhere on the
    page. Only values holding a single separator are treated this way: with
    two, as in "2.724,23", the first is a thousands mark and swapping it
    would change what the number says.

    Args:
        value: One expected value, as printed on the document.

    Returns:
        The forms that count as having read this value.
    """
    if value.count(".") + value.count(",") != 1:
        return {value}
    return {value.replace(".", ","), value.replace(",", ".")}


def missing_values(transcript: str, expected: list) -> list:
    """Return the expected values the transcript failed to reproduce.

    Comparison is by substring after whitespace normalisation: the transcript
    may wrap lines or add a unit after a figure, neither of which is an error.

    Args:
        transcript: The model's verbatim reading.
        expected: Values that are printed on the document.

    Returns:
        The values not found, in the order given.
    """
    flattened = normalise(transcript)
    return [
        value
        for value in expected
        if not any(
            normalise(variant) in flattened for variant in separator_variants(value)
        )
    ]


# A model that cannot read a table sometimes latches onto one row and emits
# it over and over with the numbers walking upward, until its token ceiling
# stops it. That is a different failure from "missed some values" and deserves
# to be named, not counted.
REPEAT_LIMIT = 20
REPEAT_SHARE = 0.3


def looks_degenerate(transcript: str) -> bool:
    """True when the transcript collapsed into one line repeated endlessly.

    Lines are compared by shape, with every run of digits replaced by "#", so
    that "Obracun za: 578-579" and "Obracun za: 579-580" count as the same
    line. Real documents do repeat a row shape, so a collapse is only called
    when one shape both repeats a lot and dominates the whole output.

    Args:
        transcript: The model's verbatim reading.

    Returns:
        True if the reading looks like a repetition loop rather than a
        transcript.
    """
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if len(lines) < REPEAT_LIMIT:
        return False
    shapes = Counter(re.sub(r"\d+", "#", line) for line in lines)
    repeats = shapes.most_common(1)[0][1]
    return repeats >= REPEAT_LIMIT and repeats / len(lines) > REPEAT_SHARE


def effective_size(width: int, height: int) -> tuple:
    """Return the pixel size the API will actually look at, at high detail.

    Args:
        width: The image's real width in pixels.
        height: The image's real height in pixels.

    Returns:
        The (width, height) after the API's own downscaling.
    """
    scale = min(MAX_SQUARE / max(width, height), 1.0)
    width, height = width * scale, height * scale
    scale = min(SHORTEST_SIDE / min(width, height), 1.0)
    return round(width * scale), round(height * scale)


def report_resolution(path: Path) -> None:
    """Print what the API will see, versus what the file holds."""
    try:
        from PIL import Image
    except ImportError:
        print("(install Pillow to see the resolution report)")
        return

    with Image.open(path) as image:
        width, height = image.size
    seen_width, seen_height = effective_size(width, height)
    kept = (seen_width * seen_height) / (width * height)
    print(f"file:  {width}x{height} ({width * height / 1e6:.1f} MP)")
    print(
        f"model: {seen_width}x{seen_height} "
        f"({seen_width * seen_height / 1e6:.1f} MP, {kept:.0%} of the pixels)"
    )


async def run_probe(image_url: str, setting: tuple, expected: list) -> None:
    """Transcribe one image at one (detail, model) setting, report the misses."""
    detail, model = setting
    print(f"\n{'=' * 70}\ndetail={detail}  model={model}\n{'=' * 70}")
    transcript = await transcribe(build_transcription_chain(detail, model), image_url)
    print(transcript)

    if looks_degenerate(transcript):
        print("\nWARNING: the transcript collapsed into a repeated line")

    if not expected:
        return
    missing = missing_values(transcript, expected)
    found = len(expected) - len(missing)
    print(f"\nread {found}/{len(expected)} expected values")
    for value in missing:
        print(f"  MISSING: {value}")


def parse_args() -> argparse.Namespace:
    """Read the image path and the settings to compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to the photo")
    parser.add_argument(
        "--expect", type=Path, help="file of values printed on the document"
    )
    parser.add_argument(
        "--detail",
        default=DEFAULT_DETAILS,
        help="comma-separated detail settings, e.g. low,high,original",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help="vision model to call; detail=original needs a newer one",
    )
    return parser.parse_args()


async def main() -> None:
    """Transcribe one image once per detail setting and report the misses."""
    args = parse_args()
    details = [setting.strip() for setting in args.detail.split(",")]
    expected = load_expected(args.expect) if args.expect else []

    report_resolution(args.image)
    print(f"\nabout to make {len(details)} real API call(s), one per setting")

    mime_type = mimetypes.guess_type(args.image.name)[0] or "image/jpeg"
    image_url = encode_image_as_data_url(args.image.read_bytes(), mime_type)
    for detail in details:
        await run_probe(image_url, (detail, args.model), expected)


if __name__ == "__main__":
    asyncio.run(main())
