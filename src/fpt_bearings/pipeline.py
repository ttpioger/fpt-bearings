import logging
from pathlib import Path

import pandas as pd

from fpt_bearings.detector import FPTDetector
from fpt_bearings.features import FeatureExtractor
from fpt_bearings.indicators import HealthIndicator
from fpt_bearings.loaders import BearingLoader
from fpt_bearings.report import ReportRow, Reporter
from fpt_bearings.smoothing import NoSmoother, Smoother
from fpt_bearings.storage import ArtifactStore

logger = logging.getLogger(__name__)


class FPTPipeline:
    """Orchestrator: wires Loader → Smoother → Indicator → Detector →
    FeatureExtractor → ArtifactStore → Reporter for a single dataset.

    All dataset-specific knowledge lives in `loader` (file format, sampling
    rate, time conversion, healthy points). This class has no `if dataset_name`
    branches — adding a new case study means constructing this with a
    different loader, nothing else.
    """

    def __init__(
            self,
            loader: BearingLoader,
            indicator: HealthIndicator,
            detector: FPTDetector,
            extractor: FeatureExtractor,
            store: ArtifactStore,
            reporter: Reporter,
            smoother: Smoother | None = None,
            overwrite: bool = False,
    ):
        self.loader = loader
        self.indicator = indicator
        self.detector = detector
        self.extractor = extractor
        self.store = store
        self.reporter = reporter
        self.smoother = smoother if smoother is not None else NoSmoother()
        self.overwrite = overwrite

    # --------------------------------------------------------------------- #
    # Public entry point
    # --------------------------------------------------------------------- #

    def run(self, original_data_path: Path) -> None:
        original_data_path = Path(original_data_path)
        if not original_data_path.is_dir():
            raise NotADirectoryError(original_data_path)

        available = {
            p.name.lower(): p
            for p in original_data_path.iterdir()
            if p.is_dir()
        }

        rows: list[ReportRow] = []
        for name in self.loader.healthy_points:
            bearing_dir = available.get(name)
            if bearing_dir is None:
                logger.warning(
                    "No data folder for '%s' under %s — skipping.",
                    name, original_data_path,
                )
                continue
            rows.append(self._process_bearing(name, bearing_dir))

        # Inverse direction: folders on disk that nobody configured.
        unconfigured = sorted(set(available) - set(self.loader.healthy_points))
        for name in unconfigured:
            logger.warning(
                "Data folder for '%s' has no healthy_points entry — skipping.",
                name,
            )

        self.reporter.write(rows)

    # --------------------------------------------------------------------- #
    # Per-bearing
    # --------------------------------------------------------------------- #

    def _process_bearing(self, name: str, bearing_dir: Path) -> ReportRow:
        logger.info("Processing bearing %s", name)

        df = self.loader.load(bearing_dir)
        df["Vibration"] = df["Vibration"].apply(self.smoother)
        df["Indicator"] = df["Vibration"].apply(self.indicator)

        healthy_point = self.loader.healthy_points[name]
        fpt_idx, found = self.detector.detect(df["Indicator"], healthy_point)
        if not found:
            logger.info(
                "No FPT detected for %s, falling back to healthy_point=%d.",
                name, healthy_point,
            )

        self._save_artifacts(name, df, fpt_idx)

        mps = self.loader.minutes_per_sample
        return ReportRow(
            bearing_name=name,
            fpt_found=found,
            healthy_point_minutes=healthy_point * mps,
            fpt_at_minutes=fpt_idx * mps,
            total_minutes=len(df) * mps,
        )

    def _save_artifacts(self, name: str, df: pd.DataFrame, fpt_idx: int) -> None:
        needs_features = self.overwrite or not self.store.has_features(name)
        needs_vibration = self.overwrite or not self.store.has_vibration(name)

        if not (needs_features or needs_vibration):
            logger.info("Features and vibration already exist for %s, skipping save.", name)
            return

        signals_after_fpt = df["Vibration"].iloc[fpt_idx:].tolist()
        if needs_features:
            features_df = self.extractor.extract(signals_after_fpt)
            self.store.save_features(name, features_df)
        if needs_vibration:
            self.store.save_vibration(name, signals_after_fpt)
