import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class ArtifactStore(Protocol):
    def has_features(self, bearing_name: str) -> bool: ...

    def has_vibration(self, bearing_name: str) -> bool: ...

    def save_features(self, bearing_name: str, features: pd.DataFrame) -> None: ...

    def save_vibration(self, bearing_name: str, vibrations: list[np.ndarray]) -> None: ...


class NpyArtifactStore:
    """Save features (.npy of a 2D array) and vibration (.npy of a 1D object
    array of signals) under <root>/features/ and <root>/vibration/.

    `root` is typically `<bearing_dataset>/<dataset_name>` — i.e. one store
    per dataset, owned by the pipeline that processes it.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    # ----- existence ------------------------------------------------------ #

    def has_features(self, bearing_name: str) -> bool:
        return self._features_path(bearing_name).is_file()

    def has_vibration(self, bearing_name: str) -> bool:
        return self._vibration_path(bearing_name).is_file()

    # ----- save ----------------------------------------------------------- #

    def save_features(self, bearing_name: str, features: pd.DataFrame) -> None:
        path = self._features_path(bearing_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features.to_numpy())
        logger.info("Features saved -> %s", path)

    def save_vibration(self, bearing_name: str, vibrations: list[np.ndarray]) -> None:
        path = self._vibration_path(bearing_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve heterogeneous-length signals via a 1-D object array,
        # matching the on-disk format your downstream code already expects.
        arr = np.empty(len(vibrations), dtype=object)
        for i, v in enumerate(vibrations):
            arr[i] = v
        np.save(path, arr, allow_pickle=True)
        logger.info("Vibration saved -> %s", path)

        # ----- internals ------------------------------------------------------ #

    def _features_path(self, name: str) -> Path:
        return self.root / "features" / f"{name}_feature.npy"

    def _vibration_path(self, name: str) -> Path:
        return self.root / "vibration" / f"{name}_vib.npy"
