import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_DEFAULT_PROJECTS = Path("/home/symphony/PycharmProjects")


def arrays_close(old, new, *, atol: float) -> tuple[bool, str]:
    """Handles both 2D float arrays (features) and 1D object arrays (vibration)."""
    if old.dtype == object or new.dtype == object:
        if len(old) != len(new):
            return False, f"len {len(old)} vs {len(new)}"
        max_diff = 0.0
        for o, n in zip(old, new):
            if o.shape != n.shape:
                return False, f"shape {o.shape} vs {n.shape}"
            max_diff = max(max_diff, float(np.abs(o - n).max()))
        return max_diff <= atol, f"max_abs_diff={max_diff:.3e}"

    if old.shape != new.shape:
        return False, f"shape {old.shape} vs {new.shape}"
    max_diff = float(np.abs(old - new).max())
    return max_diff <= atol, f"max_abs_diff={max_diff:.3e}"


def compare_npy_dir(
        label: str,
        legacy_dir: Path,
        new_dir: Path,
        *,
        allow_pickle: bool,
        atol: float,
) -> bool:
    print(f"\n=== {label} ===")
    if not legacy_dir.is_dir():
        print(f"  legacy dir not found: {legacy_dir}")
        return False
    if not new_dir.is_dir():
        print(f"  refactor dir not found: {new_dir}")
        return False

    legacy_files = sorted(legacy_dir.glob("*.npy"))
    if not legacy_files:
        print(f"  no .npy files at {legacy_dir}")
        return False

    all_match = True
    for legacy_path in legacy_files:
        new_path = new_dir / legacy_path.name
        if not new_path.is_file():
            print(f"  MISSING new: {legacy_path.name}")
            all_match = False
            continue

        old = np.load(legacy_path, allow_pickle=allow_pickle)
        new = np.load(new_path, allow_pickle=allow_pickle)
        ok, info = arrays_close(old, new, atol=atol)
        status = " OK " if ok else "DIFF"
        print(f"  {status}  {legacy_path.name:30s}  {info}")
        all_match &= ok

    return all_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["pronostia", "xjtu"])
    parser.add_argument(
        "--legacy-root", type=Path,
        default=_DEFAULT_PROJECTS / "bearing_dataset",
        help="Parent of legacy <dataset>/{features,vibration} (default: %(default)s).",
    )
    parser.add_argument(
        "--refactor-root", type=Path,
        default=_DEFAULT_PROJECTS / "bearing_dataset_refactor",
        help="Parent of refactor <dataset>/{features,vibration} (default: %(default)s).",
    )
    parser.add_argument("--skip-vibration", action="store_true",
                        help="Compare only features (faster).")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="Absolute tolerance for numerical equality.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    legacy_dir = args.legacy_root / args.dataset
    refactor_dir = args.refactor_root / args.dataset

    print(f"Legacy:   {legacy_dir}")
    print(f"Refactor: {refactor_dir}")

    features_ok = compare_npy_dir(
        "Features",
        legacy_dir / "features",
        refactor_dir / "features",
        allow_pickle=False,
        atol=args.atol,
    )

    vibration_ok = True
    if not args.skip_vibration:
        vibration_ok = compare_npy_dir(
            "Vibration",
            legacy_dir / "vibration",
            refactor_dir / "vibration",
            allow_pickle=True,
            atol=args.atol,
        )

    print()
    if features_ok and vibration_ok:
        print("Outputs match within tolerance.")
        return 0
    print("MISMATCHES detected — see lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
