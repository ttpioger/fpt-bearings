# First Prediction Time (FPT) for bearings vibration data

A Python Library for FPT detection in rotating machinery bearings. Provides a modular, dataset pipeline for loading raw vibration signals, computing an indicator (currently only Root Mean Square RMS), detecting the beginning of the degradation stage, extracting features, and saving the features and vibration in NPY files.  Inspired by the FPT proposed in [Physics guided neural network: Remaining useful life prediction of rolling bearings using long short-term memory network through dynamic weighting of degradation process](https://www.sciencedirect.com/science/article/pii/S0952197623015348).

---

## Features
- Dataset-pipeline: supports PRONOSTIA and XJTU datasets; can be extended with different case studies
- Pluggable components: Swapping detectors (default three sigma interval), smoothers (default no smoothing), indicators (default RMS), and reporters via Protocols
- Three-sigma FPT detection: Degradation detection with configurable sensitivity. The number of standard deviations for the interval bounds and the consecutive samples that needs to exceed the bounds before the FPT is declared are configurable.
- Feature extraction: time domain (RMS, peak, std, Square Root Amplitude) and frequency-domain (P7, P8) features.
- Storage: Saves features and vibration signals as .npy files for ML workflows. 
- TOML-based configuration: dataset configs for PRONOSTIA and XJTU ship with the package. Can be override with your own file.
---
## Installations
``pip install fpt-bearings ``
