import numpy as np
import pandas as pd

from fpt_bearings.storage import NpyArtifactStore


def test_round_trip_features(tmp_path):
    store = NpyArtifactStore(tmp_path)
    df = pd.DataFrame({"std": [1.0, 2.0], "rms": [0.5, 0.6]})

    assert store.has_features("bearing1_1") is False
    store.save_features("bearing1_1", df)
    assert store.has_features("bearing1_1") is True

    loaded = np.load(tmp_path / "features" / "bearing1_1_feature.npy")
    assert loaded.shape == (2, 2)
    assert loaded[0, 0] == 1.0


def test_round_trip_vibration(tmp_path):
    store = NpyArtifactStore(tmp_path)
    sigs = [np.arange(5, dtype=float), np.arange(7, dtype=float)]  # ragged

    assert store.has_vibration("bearing1_1") is False
    store.save_vibration("bearing1_1", sigs)
    assert store.has_vibration("bearing1_1") is True

    loaded = np.load(
        tmp_path / "vibration" / "bearing1_1_vib.npy", allow_pickle=True,
    )
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0], sigs[0])
    np.testing.assert_array_equal(loaded[1], sigs[1])