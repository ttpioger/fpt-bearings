from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FPTDetector(Protocol):
    def detect(self, indicator: pd.Series, healthy_point: int) -> tuple[int, bool]: ...


class _RunningStats:
    """Welford's online mean/variance — O(1) per update, numerically stable."""

    __slots__ = ("n", "mean", "_m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        # ddof=1, matches pandas Series.std()
        return (self._m2 / (self.n - 1)) ** 0.5 if self.n > 1 else float("nan")


class ThreeSigmaDetector:
    def __init__(self, k: float = 3.0, consecutive: int = 3):
        self.k = k
        self.consecutive = consecutive

    def detect(self, indicator: pd.Series, healthy_point: int) -> tuple[int, bool]:
        initial_healthy_point = healthy_point

        stats = _RunningStats()
        for v in indicator.iloc[:healthy_point]:
            stats.add(float(v))
        lower, upper = self._bounds(stats)

        triggers = 0
        for i in range(healthy_point, len(indicator)):
            value = float(indicator.iloc[i])
            if lower <= value <= upper:
                stats.add(value)
                lower, upper = self._bounds(stats)
                triggers = 0
            else:
                triggers += 1

            if triggers >= self.consecutive:
                return i - (self.consecutive - 1), True

        return initial_healthy_point, False

    def _bounds(self, stats: _RunningStats) -> tuple[float, float]:
        return stats.mean - self.k * stats.std, stats.mean + self.k * stats.std
