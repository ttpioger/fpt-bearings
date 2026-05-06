from fpt_bearings.report import ReportRow, TextReporter


def test_writes_expected_structure(tmp_path):
    out = tmp_path / "fpt_report.txt"
    reporter = TextReporter(out, title="PRONOSTIA")

    rows = [
        ReportRow("bearing1_1", True, 16.67, 120.50, 453.83),
        ReportRow("bearing1_2", False, 16.67, 16.67, 200.00),
    ]
    reporter.write(rows)

    content = out.read_text(encoding="utf-8")
    assert "FPT Report — PRONOSTIA" in content
    assert "Generated:" in content
    assert "bearing1_1" in content
    assert "bearing1_2" in content
    assert "120.50" in content
    assert "N/A (fallback)" in content  # the not-found row

    # Header + 2 separators + row count = 5 + 2 = 7 lines body, plus title/Generated
    assert content.count("\n") >= 8


def test_handles_zero_total_minutes(tmp_path):
    out = tmp_path / "report.txt"
    TextReporter(out, "TEST").write([
        ReportRow("test", False, 0.0, 0.0, 0.0),
    ])
    assert "0.00%" in out.read_text(encoding="utf-8")  # fallback %healthy = 0


def test_creates_parent_directory(tmp_path):
    out = tmp_path / "deep" / "nested" / "report.txt"  # parent doesn't exist
    TextReporter(out, "X").write([])
    assert out.is_file()