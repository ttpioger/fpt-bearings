import argparse
import logging
import time
from pathlib import Path

from fpt_bearings.detector import ThreeSigmaDetector
from fpt_bearings.features import FeatureExtractor, default_features
from fpt_bearings.indicators import RMS
from fpt_bearings.loaders import PronostiaLoader, XjtuLoader
from fpt_bearings.pipeline import FPTPipeline
from fpt_bearings.report import TextReporter
from fpt_bearings.smoothing import ExponentialSmoother, NoSmoother
from fpt_bearings.storage import NpyArtifactStore

_DEFAULT_PROJECTS = Path("/home/symphony/PycharmProjects")


def build_pipeline(dataset: str,
                   output_root: Path,
                   smoothing: bool,
                   overwrite: bool,) -> FPTPipeline:
    loader = {"pronostia": PronostiaLoader, "xjtu": XjtuLoader}[dataset]()
    output_root.mkdir(parents=True, exist_ok=True)
    return FPTPipeline(
        loader=loader,
        smoother=ExponentialSmoother(alpha=0.5) if smoothing else NoSmoother(),
        indicator=RMS(),
        detector=ThreeSigmaDetector(),
        extractor=FeatureExtractor(default_features(loader.sample_freq)),
        store=NpyArtifactStore(output_root),
        reporter=TextReporter(output_root / "fpt_report.txt", title=dataset.upper()),
        overwrite=overwrite,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["pronostia", "xjtu"])
    parser.add_argument(
        "--input-root", type=Path,
        default=_DEFAULT_PROJECTS / "bearing_dataset",
        help="Parent containing <dataset>/original_data (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_DEFAULT_PROJECTS / "bearing_dataset_refactor",
        help="Parent where the pipeline writes its outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--no-smoothing", action="store_true",
        help="Disable exponential smoothing (use NoSmoother).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite features/vibration files even if they already exist.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_dataset = args.input_root / args.dataset
    output_dataset = args.output_root / args.dataset

    print(f"Reading input from:  {input_dataset / 'original_data'}")
    print(f"Writing outputs to:  {output_dataset}")

    pipeline = build_pipeline(args.dataset, output_dataset,
                              smoothing=not args.no_smoothing,overwrite=args.overwrite)

    t0 = time.perf_counter()
    pipeline.run(input_dataset / "original_data")
    print(f"Done in {time.perf_counter() - t0:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
