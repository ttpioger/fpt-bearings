import numpy as np

from fpt_bearings.loaders import PronostiaLoader
from fpt_bearings.features import (
    FeatureExtractor, default_features, P7, RootMeanSquare,
)


def test_sample_freq_plumbed_from_loader():
    loader = PronostiaLoader()
    feats = default_features(loader.sample_freq)
    p7 = next(f for f in feats if f.name == "p7")
    assert p7.sample_freq == loader.sample_freq


def test_extract_produces_expected_columns():
    loader = PronostiaLoader()
    extractor = FeatureExtractor(default_features(loader.sample_freq))
    signals = [np.random.default_rng(0).normal(size=2560) for _ in range(3)]
    df = extractor.extract(signals)
    assert df.shape == (3, 6)
    assert df.columns.tolist() == [
        "std", "square_root_amplitude", "root_mean_square", "peak", "p7", "p8",
    ]
    assert df.notna().all().all()


def test_p7_requires_sample_freq():
    import pytest
    with pytest.raises(TypeError):
        P7()  # the strictness we wanted


def test_subset_extractor():
    extractor = FeatureExtractor([RootMeanSquare()])
    df = extractor.extract([np.array([3.0, 4.0])])  # rms = sqrt((9+16)/2) = 3.535...
    assert df.columns.tolist() == ["root_mean_square"]
    assert abs(df.iloc[0, 0] - (12.5 ** 0.5)) < 1e-12  