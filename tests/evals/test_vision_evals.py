"""Measure what the vision model can actually read from a real bill.

These send REAL requests to OpenAI and cost money, so the `eval` marker keeps
them out of every ordinary run (see pyproject.toml). Run them with:

    pytest -m eval tests/evals/test_vision_evals.py -v

The document agent reads and reasons in one pass, so a wrong answer gives no
way to tell a misread digit from a bad inference. These evals split the two
apart: they ask only for a verbatim transcript, then check it against values
that were verified against each bill's own arithmetic (see the .expected.txt
files next to the images).

Two tests per bill, deliberately graded. The headline total is large print
and should always be read; the full list includes small print inside dense
tables. If the first passes and the second fails, the model can see the page
but not the table, which is a resolution problem rather than a prompt one.

One API call per bill: transcripts are cached, so adding a test that reads an
already-transcribed bill is free.

The model and detail level are read from the VISION_MODEL and VISION_DETAIL
environment variables, so the same fixed expectations can score a different
model without touching this file:

    $env:VISION_MODEL="gpt-5.4"; $env:VISION_DETAIL="original"
    pytest -m eval tests/evals/test_vision_evals.py -v -s
"""

import os
from pathlib import Path

import pytest

from scripts.probe_vision import load_expected, looks_degenerate, missing_values
from src.agents.multimodal_agent import (
    IMAGE_DETAIL,
    LLM_MODEL,
    build_transcription_chain,
    encode_image_as_data_url,
    transcribe,
)

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="needs a real OpenAI API key"
    ),
]

ASSETS = Path(__file__).parent.parent / "assets"

# Enough of a transcript to see what went wrong, without a runaway reading
# scrolling the real failure off the terminal.
PREVIEW_CHARS = 2000

# Overridable so the same three bills can be scored against a different model
# without editing code, which is the whole point of having a fixed set of
# expected values. detail="original" needs gpt-5.4 or newer; on gpt-4o the
# API rejects it.
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)
VISION_DETAIL = os.getenv("VISION_DETAIL", IMAGE_DETAIL)

# Each bill and the one figure a reader actually came for. Large print on all
# three, so a miss here means the image never arrived in a readable state.
BILL_TOTALS = {
    "EPS_Bill_1.jpg": "1.438,49",
    "EPS_Bill_2.jpg": "7.917,25",
    "Infostan_Bill_1.JPG": "2.066,98",
}

_transcripts: dict = {}


async def read_bill(image_name: str) -> str:
    """Transcribe one bill, once per session however many tests ask for it.

    Args:
        image_name: File name inside tests/assets.

    Returns:
        The model's verbatim transcript of that image.
    """
    if image_name not in _transcripts:
        image = ASSETS / image_name
        image_url = encode_image_as_data_url(image.read_bytes(), "image/jpeg")
        chain = build_transcription_chain(VISION_DETAIL, VISION_MODEL)
        transcript = await transcribe(chain, image_url)
        # Printed, not logged: pytest captures stdout and replays it when a
        # test fails, so a failure arrives with the reading that caused it.
        # The settings are printed with it, because comparing two runs is
        # meaningless without knowing which model produced each.
        print(
            f"\n----- {image_name} | model={VISION_MODEL} "
            f"detail={VISION_DETAIL} | {len(transcript)} chars -----"
        )
        print(transcript[:PREVIEW_CHARS])
        if len(transcript) > PREVIEW_CHARS:
            print(f"... [{len(transcript) - PREVIEW_CHARS} more characters]")
        _transcripts[image_name] = transcript
    return _transcripts[image_name]


def expected_for(image_name: str) -> list:
    """Load the verified values for one bill."""
    return load_expected(ASSETS / f"{Path(image_name).stem}.expected.txt")


@pytest.mark.parametrize("image_name", sorted(BILL_TOTALS))
async def test_the_transcript_does_not_degenerate(image_name):
    # Measured on Infostan_Bill_1: unable to read the table, the model emitted
    # "Обрачун за: 578-579", then "579-580", and kept counting for some 1500
    # lines until its token ceiling stopped it. Checked first because every
    # other failure on that bill is downstream of this one.
    transcript = await read_bill(image_name)

    assert not looks_degenerate(transcript)


@pytest.mark.parametrize("image_name", sorted(BILL_TOTALS))
async def test_the_headline_total_is_read(image_name):
    # The figure the user opened the bill for, printed large. Failing this
    # means the photo never reached the model in a usable state at all.
    transcript = await read_bill(image_name)

    assert BILL_TOTALS[image_name] in transcript


@pytest.mark.parametrize("image_name", sorted(BILL_TOTALS))
async def test_every_verified_value_is_read(image_name):
    # The specification: everything a correct answer might need, including
    # the small print. What this misses is exactly what the agent cannot
    # reason about, so the failure message lists it rather than just counting.
    transcript = await read_bill(image_name)

    assert missing_values(transcript, expected_for(image_name)) == []
