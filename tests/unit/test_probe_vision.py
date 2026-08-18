from scripts.probe_vision import (
    effective_size,
    load_expected,
    looks_degenerate,
    missing_values,
    normalise,
    separator_variants,
)


def test_effective_size_matches_the_documented_downscale():
    # The measured cause of the misread digits. A 12 MP phone photo is fitted
    # into 2048x2048 and then shrunk until its shortest side is 768, so the
    # model looks at 6% of the pixels sent even at detail="high".
    assert effective_size(4032, 3024) == (1024, 768)


def test_effective_size_caps_a_large_square_image():
    assert effective_size(4000, 4000) == (768, 768)


def test_effective_size_never_upscales_a_small_image():
    # Both steps clamp their scale at 1.0, so a small photo passes untouched.
    assert effective_size(400, 300) == (400, 300)


def test_missing_values_reports_what_the_transcript_lacks():
    transcript = "Cena 8,21 RSD po kWh"

    assert missing_values(transcript, ["8,21", "521,97"]) == ["521,97"]


def test_missing_values_is_empty_when_everything_was_read():
    assert missing_values("0,00 kWh and 2.066,98 RSD", ["0,00", "2.066,98"]) == []


def test_missing_values_ignores_a_line_break_inside_the_value():
    # The model wraps long table rows; a wrapped figure was still read
    # correctly and must not be counted as a miss.
    assert missing_values("Instalisana\nsnaga 3,96 kW", ["Instalisana snaga"]) == []


def test_missing_values_does_not_match_a_value_that_was_altered():
    # The failure this harness exists to catch: 0,00 came back as 0,06.
    assert missing_values("kolicina 0,06 kWh", ["0,00"]) == ["0,00"]


def test_normalise_collapses_runs_of_whitespace():
    assert normalise("a  \n b\tc") == "a b c"


def test_normalise_rewrites_a_unicode_minus_as_a_hyphen():
    # Models write U+2212 where the bill prints a plain hyphen. That is
    # transcription style, not a misreading of the number.
    assert normalise("−252,05") == "-252,05"


def test_normalise_closes_a_gap_between_the_sign_and_its_digits():
    assert normalise("- 7.110,87") == "-7.110,87"


def test_missing_values_accepts_a_differently_written_minus():
    assert missing_values("popust –252,05 din", ["-252,05"]) == []


def test_missing_values_rejects_a_dropped_minus_sign():
    # The sign carries the meaning: -252,05 is a discount, 252,05 a charge.
    # Losing it is a misreading of the bill, so it has to fail.
    assert missing_values("popust 252,05 din", ["-252,05"]) == ["-252,05"]


def test_load_expected_keeps_only_real_values(tmp_path):
    expected_file = tmp_path / "expected.txt"
    expected_file.write_text(
        "# the bill's own figures\n8,21\n\n   # indented note\n  521,97  \n",
        encoding="utf-8",
    )

    assert load_expected(expected_file) == ["8,21", "521,97"]


def test_looks_degenerate_catches_a_repetition_loop():
    # The measured Infostan failure: unable to read the table, the model
    # emitted one row shape over and over with the numbers walking upward.
    loop = "\n".join(f"Обрачун за: {n}-{n + 1}" for n in range(578, 700))

    assert looks_degenerate(loop) is True


def test_looks_degenerate_accepts_a_real_transcript():
    # A bill legitimately repeats row shapes; only a collapse should be named.
    transcript = "\n".join(
        [
            "Рачун за електричну енергију - СЕПТЕМБАР 2025.",
            "1 Обрачунска снага (kW) 17,25 54,2580 935,95",
            "2 Трошак гарантованог снабдевача 160,67",
            "3 Енергија (kWh) 0 0,00",
            "4 ЗАДУЖЕЊЕ ЗА ЕЛЕКТРИЧНУ ЕНЕРГИЈУ 1.096,62",
        ]
    )

    assert looks_degenerate(transcript) is False


def test_looks_degenerate_ignores_a_short_reading():
    # Too little output to tell a loop from a document; never guess.
    assert looks_degenerate("Рачун\nУкупно 2.066,98") is False


def test_missing_values_accepts_a_swapped_decimal_separator():
    # Measured: the bill prints "+12.58" under its chart, and the model wrote
    # "+12,58", matching the comma used everywhere else on the page. Same
    # number, different notation.
    assert missing_values("раст +12,58 %", ["+12.58"]) == []


def test_missing_values_keeps_a_thousands_separator_strict():
    # Two separators means the first is a thousands mark. Swapping it would
    # turn 2.724,23 into a different number, so it must not be forgiven.
    assert missing_values("износе: 2,724.23 динара", ["2.724,23"]) == ["2.724,23"]


def test_separator_variants_leaves_a_plain_integer_alone():
    assert separator_variants("9230") == {"9230"}
