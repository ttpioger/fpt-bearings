from pathlib import Path

import numpy as np
import pandas as pd

from fpt_bearings.detector import ThreeSigmaDetector
from fpt_bearings.features import FeatureExtractor, default_features
from fpt_bearings.indicators import RMS
from fpt_bearings.pipeline import FPTPipeline
from fpt_bearings.report import TextReporter
from fpt_bearings.smoothing import NoSmoother
from fpt_bearings.storage import NpyArtifactStore


class _StubLoader:
    """Fake loader returning hand-built dataframes. Satisfies BearingLoader."""
    minutes_per_sample = 1.0
    sample_freq = 25_600.0
    healthy_points = {"bearing1_1": 10}

    def load(self, bearing_path: Path) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        healthy = [rng.normal(1.0, 0.05, 100) for _ in range(15)]
        broken = [rng.normal(5.0, 0.50, 100) for _ in range(5)]
        return pd.DataFrame({
            "Date": np.arange(20),
            "Vibration": healthy + broken,
        })


def test_pipeline_end_to_end(tmp_path):
    # arrange a minimal "original_data/bearing1_1" directory
    bearing_dir = tmp_path / "data" / "bearing1_1"
    bearing_dir.mkdir(parents=True)

    loader = _StubLoader()
    pipeline = FPTPipeline(
        loader=loader,
        smoother=NoSmoother(),
        indicator=RMS(),
        detector=ThreeSigmaDetector(),
        extractor=FeatureExtractor(default_features(loader.sample_freq)),
        store=NpyArtifactStore(tmp_path),
        reporter=TextReporter(tmp_path / "fpt_report.txt", title="STUB"),
    )

    pipeline.run(tmp_path / "data")

    assert (tmp_path / "fpt_report.txt").is_file()
    assert (tmp_path / "features" / "bearing1_1_feature.npy").is_file()
    assert (tmp_path / "vibration" / "bearing1_1_vib.npy").is_file()
    assert "bearing1_1" in (tmp_path / "fpt_report.txt").read_text(encoding="utf-8")
