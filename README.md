# First Prediction Time (FPT) for bearings vibration data

A Python Library for FPT detection in rotating machinery bearings. Provides a modular, dataset pipeline for loading raw vibration signals, computing an indicator (currently only Root Mean Square RMS), detecting the beginning of the degradation stage, extracting features, and saving the features and vibration in NPY files. 

## Features
- Dataset-pipeline: supports PRONOSTIA and XJTU datasets; can be extended with different case studies
- Pluggable components: Swapping detectors (default three sigma interval), smoothers (default no smoothing), indicators (default RMS), and reporters via Protocols
- 
- 
