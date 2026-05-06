from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FPTDetector(Protocol):
    def detect(self, indicator: pd.Series, healthy_point: int) -> tuple[int, bool]: ...


class ThreeSigmaDetector:
    def __init__(self, k: float = 3.0, consecutive: int = 3):
        self.k = k
        self.consecutive = consecutive

    def detect(self, indicator: pd.Series, healthy_point: int) -> tuple[int, bool]:
        initial_healthy_point = healthy_point
        lower, upper = self._bounds(indicator, healthy_point)

        triggers = 0
        for i in range(healthy_point, len(indicator)):
            value = indicator.iloc[i]
            if lower <= value <= upper:
                healthy_point += 1
                lower, upper = self._bounds(indicator, healthy_point)
                triggers = 0
            else:
                triggers += 1
            if triggers >= self.consecutive:
                return i - (self.consecutive - 1), True

        return initial_healthy_point, False

    def _bounds(self, indicator: pd.Series, n: int) -> tuple[float, float]:
        head = indicator.iloc[:n]
        m, s = head.mean(), head.std()
        return m - self.k * s, m + self.k * s