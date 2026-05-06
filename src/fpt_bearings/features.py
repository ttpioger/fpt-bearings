from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.signal import periodogram

@runtime_checkable
class Feature(Protocol):
    name: str
    def __call__(self, signal: np.ndarray) -> float: ...

# ----------------------
#      Time domain
# ----------------------

class Std:
    name = "std"
    def __call__(self, signal: np.ndarray) -> float:
        return float(np.std(signal))


class SquareRootAmplitude:
    name = "square_root_amplitude"
    def __call__(self, signal: np.ndarray) -> float:
        return float(np.mean(np.sqrt(np.abs(signal))) ** 2)


class RootMeanSquare:
    name = "root_mean_square"
    def __call__(self, signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(signal ** 2)))


class Peak:
    name = "peak"
    def __call__(self, signal: np.ndarray) -> float:
        return float(np.max(np.abs(signal)))

# ----------------------
#       Frequency
# ----------------------


_DEFAULT_KAISER_BETA = 3.0


def _spectral_spread(
        signal: np.ndarray,
        sample_freq: float,
        beta: float = _DEFAULT_KAISER_BETA,
) -> tuple[float, float]:
    """Compute (p7, center_frequency) on the detrended, Kaiser-windowed signal.

    The detrend + Kaiser-window pre-processing matches the legacy
    fpt_class.py / functions.py behavior. beta=3 is the legacy default.
    """
    detrended = (signal - np.mean(signal)) * np.kaiser(len(signal), beta)
    f, s = periodogram(detrended, sample_freq)
    fc = float(np.sum(f * s) / np.sum(s))
    p7 = float(np.sqrt(np.sum((f - fc) ** 2 * s) / len(s)))
    return p7, fc


class P7:
    name = "p7"
    def __init__(self, sample_freq: float):
        self.sample_freq = sample_freq

    def __call__(self, signal: np.ndarray) -> float:
        p7, _ = _spectral_spread(signal, self.sample_freq)
        return p7


class P8:
    name = "p8"
    def __init__(self, sample_freq: float):
        self.sample_freq = sample_freq

    def __call__(self, signal: np.ndarray) -> float:
        p7, fc = _spectral_spread(signal, self.sample_freq)
        return p7 / fc


def default_features(sample_freq: float) -> list[Feature]:
    """Standard six-feature set. Caller must provide the dataset's sampling
    rate — typically `loader.sample_freq`, which itself comes from the TOML
    config of the dataset being processed."""
    return [
        Std(),
        SquareRootAmplitude(),
        RootMeanSquare(),
        Peak(),
        P7(sample_freq),
        P8(sample_freq),
    ]


class FeatureExtractor:
    def __init__(self, features: list[Feature]):
        self.features: list[Feature] = list(features)

    def extract(self, signals: Iterable[np.ndarray]) -> pd.DataFrame:
        rows = [{f.name: f(signal) for f in self.features} for signal in signals]
        return pd.DataFrame(rows)