
from pathlib import Path
import datetime
import csv
import numpy as np
import pandas as pd
from scipy.signal import periodogram

def create_date_dict(folder: Path) -> dict:
    """
    Build a dict mapping unix timestamps to [datetime, formatted date string, filename]
    for all acc*.csv files in the given folder.
    """
    date_dict = {}
    for file_path in folder.glob("acc*.csv"):
        date_created = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

        delimiter = ';' if "Bearing1_4" in file_path.name else ','
        with file_path.open(newline="") as f:
            csv_headings = next(csv.reader(f, delimiter=delimiter))

        time_created = datetime.time(
            hour=int(float(csv_headings[0])),
            minute=int(float(csv_headings[1])),
            second=int(float(csv_headings[2])),
            microsecond=int(float(csv_headings[3])),
        )
        combined_date = datetime.datetime.combine(date_created, time_created)
        date_dict[combined_date.timestamp()] = [
            combined_date,
            combined_date.strftime("%Y-%m-%d %H:%M:%S"),
            file_path.name,
        ]
    return date_dict


def sort_xjtu_files(xjtu_path: Path) -> pd.DataFrame:
    """
    Load and concatenate all XJTU csv files, sorted by their numeric filename.
    """
    csv_files = sorted(
        xjtu_path.glob("*.csv"),
        key=lambda p: int(p.stem)
    )
    dataframes = [
        pd.read_csv(f, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'], header=0)
        for f in csv_files
    ]
    return pd.concat(dataframes, ignore_index=True)


def time_domain_features(signal: np.ndarray) -> tuple:
    """
    Extract time-domain features from a vibration signal.

    Returns: std, square_root_amplitude, root_mean_square, peak
    """
    std = np.std(signal)
    square_root_amplitude = np.square(np.mean(np.sqrt(np.abs(signal))))
    root_mean_square = np.sqrt(np.mean(np.square(signal)))
    peak = np.max(np.abs(signal))
    return std, square_root_amplitude, root_mean_square, peak


def obtain_frequency_and_psd(signal: np.ndarray, sample_freq: float = 25.6e3, beta: int = 3) -> tuple:
    """
    Compute the Power Spectral Density of the signal using a Kaiser window.
    """
    signal_detrend = signal - np.mean(signal)
    signal_detrend *= np.kaiser(len(signal_detrend), beta)
    return periodogram(signal_detrend, sample_freq)


def features_frequency_domain(signal: np.ndarray) -> tuple:
    """
    Extract frequency-domain features (p7, p8) from a vibration signal.
    """
    f, S = obtain_frequency_and_psd(signal)
    center_frequency = np.sum(f * S) / np.sum(S)
    p7 = np.sqrt(np.sum(np.square(f - center_frequency) * S) / len(S))
    p8 = p7 / center_frequency
    return p7, p8


def create_features(signal: np.ndarray) -> tuple:
    """
    Extract all time and frequency domain features from a vibration signal.

    Returns: std, square_root_amplitude, root_mean_square, peak, p7, p8
    """
    std, square_root_amplitude, root_mean_square, peak = time_domain_features(signal)
    p7, p8 = features_frequency_domain(signal)
    return std, square_root_amplitude, root_mean_square, peak, p7, p8