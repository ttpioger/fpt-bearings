import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Row schema — replaces the loose dict in fpt_class.py:272-278
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReportRow:
    bearing_name: str
    fpt_found: bool
    healthy_point_minutes: float
    fpt_at_minutes: float
    total_minutes: float


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

@runtime_checkable
class Reporter(Protocol):
    def write(self, rows: Iterable[ReportRow]) -> None: ...


# --------------------------------------------------------------------------- #
# Text reporter
# --------------------------------------------------------------------------- #

class TextReporter:
    """Fixed-width plain-text FPT report.

    Dataset-agnostic: by the time rows arrive here all values are already in
    minutes — the pipeline did the index → minutes conversion via
    `loader.minutes_per_sample`. So this class has no `if dataset_name` branches.
    """

    _COL_WIDTHS = {
        "bearing": 14,
        "tp": 12,
        "found": 10,
        "fpt_at": 24,
        "total": 20,
        "pct": 36,
    }

    def __init__(self, output_path: Path, title: str):
        self.output_path = Path(output_path)
        self.title = title

    def write(self, rows: Iterable[ReportRow]) -> None:
        rows = list(rows)
        header = self._header()
        separator = "-" * len(header)

        lines = [
            f"FPT Report — {self.title}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            separator,
            header,
            separator,
            *(self._format_row(r) for r in rows),
            separator,
        ]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("FPT report saved -> %s", self.output_path)

        # ----- internals ------------------------------------------------------ #

    def _header(self) -> str:
        w = self._COL_WIDTHS
        return (
            f"{'Bearing':<{w['bearing']}}"
            f"{'tp (min)':>{w['tp']}}"
            f"{'FPT found':>{w['found']}}"
            f"{'FPT found at (min)':>{w['fpt_at']}}"
            f"{'Total lifetime (min)':>{w['total']}}"
            f"{'% healthy stage':>{w['pct']}}"
        )

    def _format_row(self, row: ReportRow) -> str:
        w = self._COL_WIDTHS
        pct_healthy = (
            row.fpt_at_minutes / row.total_minutes * 100
            if row.total_minutes > 0 else 0.0
        )
        fpt_at_str = (
            f"{row.fpt_at_minutes:.2f}" if row.fpt_found else "N/A (fallback)"
        )
        return (
            f"{row.bearing_name:<{w['bearing']}}"
            f"{row.healthy_point_minutes:>{w['tp']}.2f}"
            f"{('True' if row.fpt_found else 'False'):>{w['found']}}"
            f"{fpt_at_str:>{w['fpt_at']}}"
            f"{row.total_minutes:>{w['total']}.2f}"
            f"{pct_healthy:>{w['pct'] - 1}.2f}%"
        )