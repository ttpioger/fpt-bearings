from typing import Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class HealthIndicator(Protocol):
    def __call__(self, signal: np.ndarray) -> float: ...


class RMS:
    def __call__(self, signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(signal ** 2)))