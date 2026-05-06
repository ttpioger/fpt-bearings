import tomllib
import csv
import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from importlib.resources import files

import numpy as np
import pandas as pd

# --------------------- config loader ---------------------
def _load_packaged_config(filename: str) -> dict:
    """Read a TOML config shipped inside the package."""
    with (files("fpt_bearings") / "configs" / filename).open('rb') as f:
        return tomllib.load(f)


def _load_external_config(path : Path) -> dict:
    """Read a TOML config from an arbitrary location on disk."""
    with path.open("rb") as f:
        return tomllib.load(f)
# ----------------------------------------------------------

@runtime_checkable
class BearingLoader(Protocol):
    """Anything with these three attributes is a valid loader."""
    minutes_per_sample: float
    sample_freq: float
    healthy_points: dict[str, int]

    def load(self, bearing_path: Path) -> pd.DataFrame:
        """ Return a DataFrame with columns ['Date','Vibration']. """
        ...


class _ConfigurableLoader:
    """Shared __init__/from_config behavior for loaders backed by a TOML config.

    Subclasses set _DEFAULT_CONFIG and implement load().
    Precedence: explicit kwarg > external config file > packaged default.
    """

    _DEFAULT_CONFIG: str  # subclasses must set this

    def __init__(
            self,
            healthy_points: dict[str, int] | None = None,
            minutes_per_sample: float | None = None,
            sample_freq = None
    ):
        cfg = _load_packaged_config(self._DEFAULT_CONFIG)
        self.healthy_points = (
            healthy_points if healthy_points is not None else cfg["healthy_points"]
        )
        self.minutes_per_sample = (
            minutes_per_sample if minutes_per_sample is not None
            else cfg["minutes_per_sample"]
        )
        self.sample_freq = (
            sample_freq if sample_freq is not None
            else cfg["sample_freq"]
        )

    @classmethod
    def from_config(cls, path: Path):
        """Build a loader from an external TOML file. Missing keys fall back to
        the packaged defaults shipped with the loader's class."""
        cfg = _load_external_config(path)
        return cls(
            healthy_points=cfg.get("healthy_points"),
            minutes_per_sample=cfg.get("minutes_per_sample"),
            sample_freq=cfg.get("sample_freq"),
        )

# --------------
#    PRONOSTIA
# --------------

class PronostiaLoader(_ConfigurableLoader):
    _DEFAULT_CONFIG = "pronostia.toml"
    _COLUMNS = ["Hour", "Minute", "Second", "µ-second", "Horiz. accel.", "vert. accel."]

    def load(self, bearing_path: Path) -> pd.DataFrame:
        index = self._build_date_index(bearing_path)
        rows = []
        for unix_ts in sorted(index):
            date_str, filename = index[unix_ts]
            df = pd.read_csv(
                bearing_path / filename,
                names=self._COLUMNS,
                sep=None,
                engine="python",
            )
            rows.append({
                "Date": date_str,
                "Vibration": df["Horiz. accel."].to_numpy(),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _build_date_index(folder: Path) -> dict[float, tuple[str, str]]:
        index: dict[float, tuple[str, str]] = {}
        for file_path in folder.glob("acc*.csv"):
            mtime_date = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).date()

            delimiter = ";" if "Bearing1_4" in file_path.name else ","
            with file_path.open(newline="") as f:
                heading = next(csv.reader(f, delimiter=delimiter))

            time_part = datetime.time(
                hour=int(float(heading[0])),
                minute=int(float(heading[1])),
                second=int(float(heading[2])),
                microsecond=int(float(heading[3])),
            )
            combined = datetime.datetime.combine(mtime_date, time_part)
            index[combined.timestamp()] = (
                combined.strftime("%Y-%m-%d %H:%M:%S"),
                file_path.name,
            )
        return index
# ----------------------
#         XJTU
# ----------------------


class XjtuLoader(_ConfigurableLoader):
    _DEFAULT_CONFIG = "xjtu.toml"
    _SAMPLES_PER_MINUTE = 32768
    _CSV_COLUMNS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]

    def load(self, bearing_path: Path) -> pd.DataFrame:
        signal = self._concat_horizontal(bearing_path)
        chunks = signal.reshape(-1, self._SAMPLES_PER_MINUTE)
        return pd.DataFrame({
            "Date": np.arange(len(chunks)),
            "Vibration": list(chunks),
        })

    def _concat_horizontal(self, bearing_path: Path) -> np.ndarray:
        csv_files = sorted(bearing_path.glob("*.csv"), key=lambda p: int(p.stem))
        frames = [pd.read_csv(f, names=self._CSV_COLUMNS, header=0) for f in csv_files]
        merged = pd.concat(frames, ignore_index=True)
        return merged["Horizontal_vibration_signals"].to_numpy()

