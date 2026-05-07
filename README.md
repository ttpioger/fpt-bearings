# First Prediction Time (FPT) for bearings vibration data

A Python Library for FPT detection in rotating machinery bearings. Provides a modular, dataset pipeline for loading raw vibration signals, computing an indicator (currently only Root Mean Square RMS), detecting the beginning of the degradation stage, extracting features, and saving the features and vibration in NPY files.  Inspired by the FPT proposed in [Physics guided neural network: Remaining useful life prediction of rolling bearings using long short-term memory network through dynamic weighting of degradation process](https://www.sciencedirect.com/science/article/pii/S0952197623015348).

---

## Features
- Dataset-pipeline: supports PRONOSTIA and XJTU datasets; can be extended with different case studies
- Pluggable components: Swapping detectors (default three sigma interval), smoothers (default no smoothing), indicators (default RMS), and reporters via Protocols
- Three-sigma FPT detection: Degradation detection with configurable sensitivity. The number of standard deviations for the interval bounds and the consecutive samples that needs to exceed the bounds before the FPT is declared are configurable.
- Feature extraction: time domain (RMS, peak, std, Square Root Amplitude) and frequency-domain (P7, P8) features.
- Storage: Saves features and vibration signals as .npy files for ML workflows. 
- TOML-based configuration: dataset configs for PRONOSTIA and XJTU ship with the package. Can be overridden with your own file.
---

## Installations

```pip install fpt-bearings ```
---

## Quick Start

```
from pathlib import Path
from fpt_bearings.loaders import PronostiaLoader
from fpt_bearings.indicators import RMS
from fpt_bearings.detector import ThreeSigmaDetector
from fpt_bearings.features import FeatureExtractor, default_features
from fpt_bearings.smoothing import ExponentialSmoother
from fpt_bearings.storage import NpyArtifactStore
from fpt_bearings.report import TextReporter
from fpt_bearings.pipeline import FPTPipeline

loader = PronostiaLoader()

pipeline = FPTPipeline(
    loader=loader,
    indicator=RMS(),
    detector=ThreeSigmaDetector(k=3.0, consecutive=3),
    extractor=FeatureExtractor(default_features(loader.sample_freq)),
    store=NpyArtifactStore(Path("output/pronostia")),
    reporter=TextReporter(Path("output/pronostia/report.txt"), title="PRONOSTIA"),
    smoother=ExponentialSmoother(alpha=0.5),
)

pipeline.run(Path("/data/PRONOSTIA/Test_set"))
```
---

## FPT overview
```
Raw vibration files
       │
       ▼
   BearingLoader        ← PronostiaLoader / XjtuLoader / custom
       │
       ▼
    Smoother            ← ExponentialSmoother / NoSmoother / custom
       │
       ▼
  HealthIndicator       ← RMS / custom
       │
       ▼
   FPTDetector          ← ThreeSigmaDetector / custom
       │
    ┌──┴──────────────┐
    ▼                 ▼
FeatureExtractor   ArtifactStore    ← features + vibration saved as .npy                      │
                      ▼
                   Reporter         ← TextReporter / custom
```

## Classes

### Loaders

** Dataset config shipped with package**:
| Class | Dataset | Config |
|---|---|---|
| `PronostiaLoader` | FEMTO / PRONOSTIA | `pronostia.toml` |
| `XjtuLoader` | XJTU-SY | `xjtu.toml` |

Load an external config

```loader = PronostiaLoader.from_config(Path("my_config.toml"))```

Example of config format (TOML): 


```
minutes_per_sample = 10.0
sample_freq = 25600.0

[healthy_points]
bearing1_1 = 5
bearing1_2 = 8
```

### Smoother
```
smoother = ExponentialSmoother(alpha=0.3)  # lower alpha = smoother
smoother = NoSmoother()                    # pass-through (default)
```

### Health Indicator
```
indicator = RMS()
```

### Detector
```python
detector = ThreeSigmaDetector(k=3.0, consecutive=3)
fpt_index, found = detector.detect(indicator_series, healthy_point=10)
```

### Feature Extractor
```python
extractor = FeatureExtractor(default_features(sample_freq=25600.0))
features_df = extractor.extract(list_of_signals)
```

### Artifact Store
```python
store = NpyArtifactStore(Path("output/pronostia"))
```

### Reporter
```python
reporter = TextReporter(Path("output/report.txt"), title="PRONOSTIA")
```


