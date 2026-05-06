from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import lfilter, lfilter_zi

@runtime_checkable
class Smoother(Protocol):
    def __call__(self, signal: np.ndarray) -> np.ndarray: ...


class ExponentialSmoother:
    """Low-pass exponential smoothing: S[i] = alpha * x[i] + (1 - alpha) * S[i-1].

    Higher alpha = less smoothing.
    """
    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._b = np.array([alpha])
        self._a = np.array([1.0, -(1.0 - alpha)])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            raise ValueError("Vibration signal is empty.")
            # Initial condition y[0] == x[0]
        zi = lfilter_zi(self._b, self._a) * signal[0]
        smoothed, _ = lfilter(self._b, self._a, signal, zi=zi)
        return smoothed


class NoSmoother:
    """Pass-through.
    Pipeline can always do `df['Vibration'].apply(self.smoother) without an Optional check."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal