from pathlib import Path

import numpy as np
import pandas as pd

from fpt_bearings.detector import ThreeSigmaDetector
from fpt_bearings.indicators import RMS
from fpt_bearings.loaders import PronostiaLoader
from fpt_bearings.smoothing import ExponentialSmoother


def naive_detect(indicator: pd.Series, healthy_point: int, k=3.0, consecutive=3):
    """Identical algorithm to your legacy get_fpt — pandas mean/std each step."""
    initial = healthy_point
    triggers = 0
    for i in range(healthy_point, len(indicator)):
        head = indicator.iloc[:healthy_point]
        m, s = head.mean(), head.std()
        lower, upper = m - k * s, m + k * s
        v = indicator.iloc[i]
        if lower <= v <= upper:
            healthy_point += 1
            triggers = 0
        else:
            triggers += 1
        if triggers >= consecutive:
            return i - (consecutive - 1), True
    return initial, False


bearing_dir = Path("/home/symphony/PycharmProjects/bearing_dataset/pronostia/original_data/Bearing1_3")

loader = PronostiaLoader()
df = loader.load(bearing_dir)
df["Vibration"] = df["Vibration"].apply(ExponentialSmoother(0.5))
df["Indicator"] = df["Vibration"].apply(RMS())

idx_welford, found_w = ThreeSigmaDetector().detect(df["Indicator"], 100)
idx_naive, found_n = naive_detect(df["Indicator"], 100)

print(f"Total rows: {len(df)}")
print(f"Welford  : idx={idx_welford} found={found_w}  → post-FPT rows={len(df) - idx_welford}")
print(f"Naive    : idx={idx_naive}   found={found_n}  → post-FPT rows={len(df) - idx_naive}")
print(f"Difference: {idx_welford - idx_naive} rows")