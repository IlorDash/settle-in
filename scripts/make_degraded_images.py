"""Build the images group E4 is measured on - unreadable, and non-document.

`thesis/assets/` holds 29 photographs of real bills, all of them readable, and
8 photographs that carry no document at all. E4 asks what the bot does with an
image it cannot read, and neither of those sets answers it on its own: a
photograph of a street is a different failure from a bill that has been
photographed too small to read. This script builds the missing source, by
damaging real bills in four named ways and by writing two frames that carry
nothing at all.

Nothing here calls the API, and nothing here is a measurement. The images are
the input to

    python -m scripts.evaluate_documents --set degraded

and the E4 entry has to say which of its figures came from which source, so
the damage must be reproducible: the same sources, the same parameters and
the same file names on every run.

Pillow is not a declared dependency of this project - `scripts/probe_vision.py`
treats it as optional - so this script says so and stops rather than adding
one.

Run from the repo root:

    python -m scripts.make_degraded_images
    python -m scripts.make_degraded_images --sources 3
"""

import argparse
from pathlib import Path

SOURCE_DIR = Path("thesis/assets")
OUTPUT_DIR = SOURCE_DIR / "degraded"

DOCUMENT_PREFIXES = ("eps_", "infostan_")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# The longest side a downscaled copy is reduced to. E0's arithmetic is the
# argument for going this far: the API's own reduction already costs a 12
# megapixel phone photograph 94% of its pixels and the bill stays readable,
# so a copy that is genuinely unreadable has to be very much smaller.
DOWNSCALED_LONGEST_SIDE = 160

# Blur radius as a share of the longest side, so a 1 megapixel scan and a 12
# megapixel photograph are damaged by the same amount rather than by the same
# number of pixels.
BLUR_SHARE = 0.01

# The corner crop keeps this share of each side, so a quarter of the width and
# a quarter of the height - one sixteenth of the page. What is left is a
# fragment of a real document rather than an unreadable one, which is a
# different question and is why it is a separate kind.
CROP_SHARE = 0.25

ROTATION_DEGREES = 180

DEGENERATE_SIZE = (1024, 1024)
WHITE = (255, 255, 255)
SOLID_COLOUR = (128, 128, 128)

# Written as PNG whatever the source was. JPEG would add compression
# artefacts of its own on top of the damage, and E4 has to be able to say
# what made an image unreadable.
OUTPUT_SUFFIX = ".png"


def downscaled(image):
    """Shrink the page until its longest side is DOWNSCALED_LONGEST_SIDE."""
    scale = DOWNSCALED_LONGEST_SIDE / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size)


def blurred(image):
    """Blur the page by a radius proportional to its size."""
    from PIL import ImageFilter

    return image.filter(ImageFilter.GaussianBlur(BLUR_SHARE * max(image.size)))


def cropped_to_a_corner(image):
    """Keep the top left corner and throw the rest of the page away."""
    return image.crop(
        (0, 0, round(image.width * CROP_SHARE), round(image.height * CROP_SHARE))
    )


def rotated(image):
    """Turn the page upside down, which is how a photograph arrives."""
    return image.rotate(ROTATION_DEGREES, expand=True)


DAMAGE = {
    "downscaled": downscaled,
    "blurred": blurred,
    "cropped": cropped_to_a_corner,
    "rotated": rotated,
}


def choose_sources(directory: Path, per_issuer: int) -> list[Path]:
    """Take the first few bills of each issuer, in a fixed order.

    Both issuers are represented, because an EPS bill and an Infostan bill
    are laid out differently and E4's question is about the image and not
    about the issuer.

    Args:
        directory: Where the photographs are.
        per_issuer: How many bills to take from each prefix.

    Returns:
        The chosen files, issuer by issuer.
    """
    chosen = []
    for prefix in DOCUMENT_PREFIXES:
        matching = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.stem.startswith(prefix)
            and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        matching.sort(key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        chosen += matching[:per_issuer]
    return chosen


def write_damaged_copies(sources: list[Path], output_dir: Path) -> list[Path]:
    """Write one copy per source per kind of damage.

    Args:
        sources: The bills to damage.
        output_dir: Where the copies go.

    Returns:
        The files written.
    """
    from PIL import Image

    written = []
    for source in sources:
        with Image.open(source) as opened:
            page = opened.convert("RGB")
        for kind, damage in DAMAGE.items():
            destination = output_dir / f"{source.stem}_{kind}{OUTPUT_SUFFIX}"
            damage(page).save(destination)
            written.append(destination)
    return written


def write_degenerate_frames(output_dir: Path) -> list[Path]:
    """Write the two frames that carry nothing at all.

    A blank white frame and a single mid-grey one. Both are valid images and
    neither holds a document, so an answer that describes one is the honest
    reply and an answer that reads a bill off one is the failure E4 looks for.

    Args:
        output_dir: Where the frames go.

    Returns:
        The files written.
    """
    from PIL import Image

    written = []
    for name, colour in (("blank_white", WHITE), ("solid_colour", SOLID_COLOUR)):
        destination = output_dir / f"{name}{OUTPUT_SUFFIX}"
        Image.new("RGB", DEGENERATE_SIZE, colour).save(destination)
        written.append(destination)
    return written


def main() -> None:
    """Write the degraded set and print what it holds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=int,
        default=1,
        help="bills to damage per issuer (default 1, so 2 bills and 8 copies)",
    )
    arguments = parser.parse_args()

    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Pillow is not installed, and it is not a dependency of this "
            "project. Source 1 of E4 - documents made unreadable on purpose - "
            "cannot be built without it, so the entry has to record that this "
            "source was skipped."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = choose_sources(SOURCE_DIR, arguments.sources)
    if not sources:
        raise SystemExit(f"no bills found in {SOURCE_DIR}")

    print(f"{len(sources)} source bill(s): {', '.join(p.name for p in sources)}")
    written = write_damaged_copies(sources, OUTPUT_DIR)
    written += write_degenerate_frames(OUTPUT_DIR)

    print(f"\n{len(written)} image(s) written to {OUTPUT_DIR.resolve()}")
    for path in written:
        print(f"  {path.name}")
    print(
        "\nsettings: "
        f"downscaled to {DOWNSCALED_LONGEST_SIDE}px on the longest side, "
        f"blurred at {BLUR_SHARE:.0%} of it, "
        f"cropped to the top {CROP_SHARE:.0%} of each side, "
        f"rotated {ROTATION_DEGREES} degrees"
    )


if __name__ == "__main__":
    main()
