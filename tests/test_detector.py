import pandas as pd

from fpt_bearings.detector import ThreeSigmaDetector

# Healthy values: mean ≈ 1.0, std ≈ 0.029 → 3-sigma band ≈ [0.91, 1.09]
HEALTHY = [0.95, 1.05, 0.98, 1.02, 1.00, 1.01, 0.99, 1.03, 0.97, 1.00]


def test_detects_first_outlier_in_run():
    in_band = [1.05, 0.95, 1.02, 0.97, 1.01]  # all within the band
    outliers = [5.0, 5.1, 5.2, 5.3, 5.4]  # all clearly outside
    series = pd.Series(HEALTHY + in_band + outliers)

    idx, found = ThreeSigmaDetector().detect(series, healthy_point=10)
    assert found is True
    assert idx == 15  # first of the three consecutive outliers


def test_falls_back_to_initial_healthy_point_when_not_found():
    no_degradation = [1.05, 0.95, 1.02]  # all within the band
    series = pd.Series(HEALTHY + no_degradation)

    idx, found = ThreeSigmaDetector().detect(series, healthy_point=10)
    assert found is False
    assert idx == 10  # initial, not the updated value