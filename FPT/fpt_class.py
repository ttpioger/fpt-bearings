from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from functions import create_date_dict, sort_xjtu_files, create_features
from fpt_constants import HEALTHY_POINTS


class FirstPredictionTime:
    def __init__(self,
                 dataset_name: str,
                 smoothing: bool = True,
                 overwrite: bool = False,
                 ):
        self.dataset_name = dataset_name.lower()
        self.smoothing = smoothing
        self.overwrite = overwrite
        self.path_to_dataset = Path(__file__).parents[2] / "bearing_dataset" / self.dataset_name
        self.healthy_points = HEALTHY_POINTS[self.dataset_name]

    # ------------------------------------------------------------------ #
    #  File checks                                                         #
    # ------------------------------------------------------------------ #

    def check_if_files_exist(self, bearing_name: str) -> str:
        save_features = self.path_to_dataset / 'features' / f'{bearing_name}_feature.npy'
        save_vibration = self.path_to_dataset / 'vibration' / f'{bearing_name}_vib.npy'

        features_exist = save_features.is_file()
        vibration_exist = save_vibration.is_file()

        if features_exist and vibration_exist and not self.overwrite:
            print(f'Both files exist for {bearing_name}, skipping.')
            return 'both'
        elif features_exist and not self.overwrite:
            print(f'Feature file already exists for {bearing_name}, only vibration will be processed.')
            return 'features_only'
        elif vibration_exist and not self.overwrite:
            print(f'Vibration file already exists for {bearing_name}, only features will be processed.')
            return 'vibration_only'
        else:
            print(f'No files exist for {bearing_name}, processing both.')
            return 'none'

    # ------------------------------------------------------------------ #
    #  Smoothing                                                           #
    # ------------------------------------------------------------------ #

    def exponential_smoothing(self, vibration: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Exponential smoothing as a low-pass filter to remove high-frequency noise.
        S[i] = alpha * x[i] + (1 - alpha) * S[i-1]
        https://en.wikipedia.org/wiki/Exponential_smoothing

        :param vibration: 1D array of vibration signal
        :param alpha: smoothing factor in (0, 1]. Higher = less smoothing.
        """
        if len(vibration) == 0:
            raise ValueError("Vibration signal is empty.")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        smoothed = np.zeros_like(vibration, dtype=np.float64)
        smoothed[0] = vibration[0]
        for i in range(1, len(vibration)):
            smoothed[i] = alpha * vibration[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    # ------------------------------------------------------------------ #
    #  Dataset loaders                                                     #
    # ------------------------------------------------------------------ #

    def load_pronostia_bearing(self, bearing_path: Path) -> pd.DataFrame:
        """Load and build the dataframe for a single PRONOSTIA bearing."""
        columns = ["Hour", "Minute", "Second", "µ-second", "Horiz. accel.", "vert. accel."]
        date_dict = create_date_dict(bearing_path)
        date_list = sorted(date_dict.keys())

        dataset_dict = {'Date': [], 'RMS': [], 'Vibration': []}

        for unix_timestamp in date_list:
            date_nice_format = date_dict[unix_timestamp][1]
            filename = date_dict[unix_timestamp][2]
            df = pd.read_csv(bearing_path / filename, names=columns, sep=None, engine='python')
            vibration = df["Horiz. accel."].to_numpy()
            if self.smoothing:
                vibration = self.exponential_smoothing(vibration)
            dataset_dict['Date'].append(date_nice_format)
            dataset_dict['RMS'].append(np.sqrt(np.mean(vibration ** 2)))
            dataset_dict['Vibration'].append(vibration)

        return pd.DataFrame(dataset_dict)

    def load_xjtu_bearing(self, bearing_path: Path) -> pd.DataFrame:
        """Load and build the dataframe for a single XJTU bearing."""
        xjtu_df = sort_xjtu_files(bearing_path)
        vibration = xjtu_df['Horizontal_vibration_signals'].to_numpy()
        if self.smoothing:
            vibration = self.exponential_smoothing(vibration)
        vibration = vibration.reshape(-1, 32768)

        time = np.arange(len(vibration))
        dataset_dict = {
            'Date': time,
            'RMS': [np.sqrt(np.mean(v ** 2)) for v in vibration],
            'Vibration': list(vibration)
        }
        return pd.DataFrame(dataset_dict)

    # ------------------------------------------------------------------ #
    #  FPT detection                                                       #
    # ------------------------------------------------------------------ #

    def get_fpt(self, dataframe: pd.DataFrame, healthy_point: int) -> tuple:
        """
        Detect the First Prediction Time using a 3-sigma judgment interval.
        The interval expands as long as points remain healthy.
        FPT is triggered after 3 consecutive out-of-bound points.

        :return: (fpt_date, not_found_fpt) where not_found_fpt=True means no FPT was detected.
        """
        not_found_fpt = True
        mean_healthy = dataframe['RMS'][:healthy_point].mean()
        std_healthy = dataframe['RMS'][:healthy_point].std()
        lower = mean_healthy - 3 * std_healthy
        upper = mean_healthy + 3 * std_healthy

        final_fpt_date = dataframe['Date'].iloc[healthy_point]
        consecutive_triggers = 0

        for i in range(healthy_point, len(dataframe)):
            rms_at_i = dataframe['RMS'].iloc[i]
            if lower <= rms_at_i <= upper:
                healthy_point += 1
                mean_healthy = dataframe['RMS'][:healthy_point].mean()
                std_healthy = dataframe['RMS'][:healthy_point].std()
                lower = mean_healthy - 3 * std_healthy
                upper = mean_healthy + 3 * std_healthy
                consecutive_triggers = 0
            else:
                consecutive_triggers += 1

            if consecutive_triggers >= 3:
                final_fpt_date = dataframe['Date'].iloc[i - 2]
                not_found_fpt = False
                break

        return final_fpt_date, not_found_fpt

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def get_features(self,
                      dataframe: pd.DataFrame,
                      final_fpt_date,
                      not_found_fpt: bool,
                      bearing_name: str,
                      healthy_point: int) -> tuple:
        """
        Extract features and vibration arrays for all samples after the FPT.
        Falls back to the configured healthy_point index if no FPT was detected.
        """
        print("----------------")
        print(f"Extracting features for {self.dataset_name} bearing: {bearing_name}")

        if not_found_fpt:
            print("No FPT detected, falling back to configured healthy point.")
            final_fpt_date = dataframe['Date'].iloc[healthy_point]

        dataset_after_fpt = dataframe.loc[dataframe['Date'] >= final_fpt_date].copy()

        print(f"Start       : {dataframe['Date'].iloc[0]}")
        print(f"FPT         : {final_fpt_date}")
        print(f"End         : {dataframe['Date'].iloc[-1]}")
        print(f"Signal length — original: {len(dataframe)}, after FPT: {len(dataset_after_fpt)}")

        feature_dict = {
            'std': [], 'square_root_amplitude': [], 'root_mean_square': [],
            'peak': [], 'p7': [], 'p8': []
        }
        vibration_list = []

        for values in dataset_after_fpt["Vibration"].to_numpy():
            std, square_root_amplitude, root_mean_square, peak, p7, p8 = create_features(values)
            feature_dict['std'].append(std)
            feature_dict['square_root_amplitude'].append(square_root_amplitude)
            feature_dict['root_mean_square'].append(root_mean_square)
            feature_dict['peak'].append(peak)
            feature_dict['p7'].append(p7)
            feature_dict['p8'].append(p8)
            vibration_list.append(values)

        feature_df = pd.DataFrame(feature_dict)
        vibration_df = pd.DataFrame({'vibration': vibration_list})
        return feature_df, vibration_df

    # ------------------------------------------------------------------ #
    #  Saving                                                              #
    # ------------------------------------------------------------------ #

    def save_features(self, feature_df: pd.DataFrame, bearing_name: str) -> None:
        save_path = self.path_to_dataset / 'features' / f'{bearing_name}_feature.npy'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, feature_df.to_numpy())
        print(f'Features saved → {save_path}')

    def save_vibration(self, vibration_df: pd.DataFrame, bearing_name: str) -> None:
        save_path = self.path_to_dataset / 'vibration' / f'{bearing_name}_vib.npy'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, vibration_df['vibration'].to_numpy())
        print(f'Vibration saved → {save_path}')

    def save(self, feature_df: pd.DataFrame, vibration_df: pd.DataFrame,
              bearing_name: str, existing: str) -> None:
        if existing == 'vibration_only':
            # Vibration exists, only recompute and save features
            print(f'Only saving missing features for {bearing_name}.')
            self.save_features(feature_df, bearing_name)
        elif existing == 'features_only':
            # Features exist, only recompute and save vibration
            print(f'Only saving missing vibration for {bearing_name}.')
            self.save_vibration(vibration_df, bearing_name)
        else:
            # Neither exists, save both
            self.save_features(feature_df, bearing_name)
            self.save_vibration(vibration_df, bearing_name)

    # ------------------------------------------------------------------ #
    #  Main process                                                        #
    # ------------------------------------------------------------------ #

    def fpt_process(self) -> None:
        original_data_path = self.path_to_dataset / 'original_data'
        loader = self.load_pronostia_bearing if self.dataset_name == 'pronostia' else self.load_xjtu_bearing
        report_rows = []

        for bearing_dir in sorted(original_data_path.iterdir()):
            if not bearing_dir.is_dir():
                continue

            bearing_name = bearing_dir.name.lower()

            if bearing_name not in self.healthy_points:
                print(f'No healthy point configured for {bearing_name}, skipping.')
                continue

            existing = self.check_if_files_exist(bearing_name)

            healthy_point = self.healthy_points[bearing_name]
            dataframe = loader(bearing_dir)
            final_fpt_date, not_found_fpt = self.get_fpt(dataframe, healthy_point)

            if existing != 'both':
                feature_df, vibration_df = self.get_features(
                    dataframe, final_fpt_date, not_found_fpt, bearing_name, healthy_point
                )
                self.save(feature_df, vibration_df, bearing_name, existing)

            # --- collect report data regardless of existing status ---
            if not_found_fpt:
                final_fpt_date = dataframe['Date'].iloc[healthy_point]

            if self.dataset_name == 'pronostia':
                healthy_point_minutes = round(healthy_point * (10 / 60), 2)
                total_minutes = round(len(dataframe) * (10 / 60), 2)
            else:
                healthy_point_minutes = float(healthy_point)
                total_minutes = float(dataframe['Date'].iloc[-1])

            report_rows.append({
                'bearing_name': bearing_name,
                'fpt_found': not not_found_fpt,
                'healthy_point_minutes': healthy_point_minutes,
                'fpt_at_minutes': self.compute_fpt_minute(dataframe, final_fpt_date),
                'total_minutes': total_minutes,
            })

        self.save_fpt_report(report_rows)

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def compute_fpt_minute(self, dataframe: pd.DataFrame, fpt_date) -> float:
        """Convert FPT date to minutes based on dataset sampling."""
        if self.dataset_name == 'pronostia':
            # PRONOSTIA samples every 10 seconds → index * (10/60) minutes
            fpt_index = dataframe[dataframe['Date'] == fpt_date].index[0]
            return round(fpt_index * (10 / 60), 2)
        else:
            # XJTU samples every 1 minute → Date column is already in minutes
            return float(fpt_date)

    def save_fpt_report(self, report_rows: list) -> None:
        """
        Save a plain-text FPT summary table to:
            bearing_dataset/<dataset>/fpt_report.txt

        Each row in report_rows is a dict with keys:
            bearing_name, fpt_found, healthy_point_minutes,
            fpt_at_minutes, total_minutes
        """
        save_path = self.path_to_dataset / 'fpt_report.txt'

        col_widths = {
            'bearing': 14,
            'tp': 12,
            'found': 10,
            'fpt_at': 24,
            'total': 20,
            'pct': 36,
        }

        header = (
            f"{'Bearing':<{col_widths['bearing']}}"
            f"{'tp (min)':>{col_widths['tp']}}"
            f"{'FPT found':>{col_widths['found']}}"
            f"{'FPT found at (min)':>{col_widths['fpt_at']}}"
            f"{'Total lifetime (min)':>{col_widths['total']}}"
            f"{'% healthy stage':>{col_widths['pct']}}"
        )
        separator = '-' * len(header)

        lines = [
            f"FPT Report — {self.dataset_name.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            separator,
            header,
            separator,
        ]

        for row in report_rows:
            pct_healthy = (row['fpt_at_minutes'] / row['total_minutes'] * 100) if row['total_minutes'] > 0 else 0.0
            fpt_at_str = f"{row['fpt_at_minutes']:.2f}" if row['fpt_found'] else 'N/A (fallback)'
            line = (
                f"{row['bearing_name']:<{col_widths['bearing']}}"
                f"{row['healthy_point_minutes']:>{col_widths['tp']}.2f}"
                f"{'True' if row['fpt_found'] else 'False':>{col_widths['found']}}"
                f"{fpt_at_str:>{col_widths['fpt_at']}}"
                f"{row['total_minutes']:>{col_widths['total']}.2f}"
                f"{pct_healthy:>{col_widths['pct'] - 1}.2f}%"
            )
            lines.append(line)

        lines.append(separator)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        print(f'FPT report saved → {save_path}')
    def run_fpt(self) -> None:
        print(f'Starting First Prediction Time (FPT) process — dataset: {self.dataset_name}')
        self.fpt_process()
        print('FPT process complete.')


if __name__ == '__main__':
    fpt = FirstPredictionTime("pronostia", overwrite=True, smoothing=True)
    fpt.run_fpt()