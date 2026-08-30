from pathlib import Path

from scripts import measure_footprint

# A cut-down /proc/self/status, in the kernel's own format.
STATUS = """Name:\tpython3
VmPeak:\t  512345 kB
VmSize:\t  498765 kB
VmHWM:\t   210496 kB
VmRSS:\t   198144 kB
Threads:\t5
"""


def test_read_memory_reads_the_resident_size_and_the_high_water_mark(
    tmp_path, monkeypatch
):
    status = tmp_path / "status"
    status.write_text(STATUS, encoding="utf-8")
    monkeypatch.setattr(measure_footprint, "STATUS_PATH", status)

    assert measure_footprint.read_memory_kb() == {
        "rss_kb": 198144,
        "peak_rss_kb": 210496,
    }


def test_read_memory_reports_nothing_where_proc_does_not_exist(monkeypatch):
    # Windows has no /proc, and the script has to keep timing the start-up
    # rather than failing on a machine that cannot weigh itself.
    monkeypatch.setattr(measure_footprint, "STATUS_PATH", Path("no/such/status"))

    assert measure_footprint.read_memory_kb() == {"rss_kb": None, "peak_rss_kb": None}


def test_read_memory_ignores_the_fields_it_was_not_asked_for(tmp_path, monkeypatch):
    # VmPeak is virtual size, not resident, and must not be read as the peak.
    status = tmp_path / "status"
    status.write_text(STATUS, encoding="utf-8")
    monkeypatch.setattr(measure_footprint, "STATUS_PATH", status)

    assert measure_footprint.read_memory_kb()["peak_rss_kb"] != 512345


def test_stage_records_the_time_and_the_memory_of_one_step(monkeypatch):
    monkeypatch.setattr(
        measure_footprint, "read_memory_kb", lambda: {"rss_kb": 1, "peak_rss_kb": 2}
    )
    stages = []

    ended = measure_footprint.stage("imports", 0.0, stages)

    assert [row["stage"] for row in stages] == ["imports"]
    assert stages[0]["ms"] > 0
    assert stages[0]["rss_kb"] == 1
    assert ended > 0
