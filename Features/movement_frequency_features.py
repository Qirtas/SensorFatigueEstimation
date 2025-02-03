import os
import glob
import re
import numpy as np
import pandas as pd

from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


###############################################################################
# Helper Functions for Frequency Features
###############################################################################

def compute_frequency_spectrum(signal, sampling_rate):
    """
    Compute the frequency spectrum of a 1D signal using real FFT.
    Returns:
      freqs : array of frequency bins (Hz)
      mag_spectrum : power at each frequency bin
    """
    n = len(signal)
    if n < 2:
        return np.array([0.0]), np.array([0.0])

    # Real FFT
    fft_vals = rfft(signal)
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    # Power spectrum
    mag_spectrum = np.abs(fft_vals) ** 2
    return freqs, mag_spectrum


def compute_statistical_freq_features(freqs, spectrum, prefix):
    """
    Statistical features of the frequency 'distribution' (mean freq, median freq, etc.)
    freq: array of frequency bins
    spectrum: power at each bin
    prefix: string prefix for naming (e.g., "X_Shoulder_acc")
    """
    feats = {}
    if len(freqs) < 2 or np.sum(spectrum) == 0:
        for f in ["MeanFrequency", "MedianFrequency", "FrequencyVariance",
                  "FrequencyRange", "FrequencyStd", "FrequencySkewness", "FrequencyKurtosis"]:
            feats[f"{prefix}_{f}"] = np.nan
        return feats

    total_power = np.sum(spectrum)

    # 1. Mean Frequency
    mean_freq = np.sum(freqs * spectrum) / total_power

    # 2. Median Frequency
    half_power = total_power / 2.0
    cumsum_power = np.cumsum(spectrum)
    median_idx = np.searchsorted(cumsum_power, half_power)
    if median_idx >= len(freqs):
        median_idx = len(freqs) - 1
    median_freq = freqs[median_idx]

    # 3. Frequency Variance
    var_freq = np.sum(spectrum * (freqs - mean_freq) ** 2) / total_power

    # 4. Frequency Range
    valid_mask = (spectrum > 1e-12)
    if np.any(valid_mask):
        freq_min = freqs[valid_mask].min()
        freq_max = freqs[valid_mask].max()
        freq_range = freq_max - freq_min
    else:
        freq_range = 0.0

    freq_std = np.sqrt(var_freq)

    # Weighted moments for skew/kurt
    p = spectrum / total_power
    m3 = np.sum(p * (freqs - mean_freq) ** 3)
    m4 = np.sum(p * (freqs - mean_freq) ** 4)
    if freq_std > 1e-12:
        freq_skew = m3 / (freq_std ** 3)
        freq_kurt = (m4 / (freq_std ** 4)) - 3.0
    else:
        freq_skew = 0.0
        freq_kurt = -3.0

    feats[f"{prefix}_MeanFrequency"] = mean_freq
    feats[f"{prefix}_MedianFrequency"] = median_freq
    feats[f"{prefix}_FrequencyVariance"] = var_freq
    feats[f"{prefix}_FrequencyRange"] = freq_range
    feats[f"{prefix}_FrequencyStd"] = freq_std
    feats[f"{prefix}_FrequencySkewness"] = freq_skew
    feats[f"{prefix}_FrequencyKurtosis"] = freq_kurt

    return feats


def compute_frequency_domain_features(freqs, spectrum, prefix):
    """
    Additional frequency-domain features:
      - Peak Frequency
      - Bandwidth (90% of power)
      - Spectral Entropy
      - Harmonic Energy Ratio
      - Energy in Low/Mid/High bands
    """
    feats = {}
    if len(freqs) < 2 or np.sum(spectrum) == 0:
        for f in ["PeakFrequency", "Bandwidth", "SpectralEntropy", "HarmonicEnergyRatio"]:
            feats[f"{prefix}_{f}"] = np.nan
        feats[f"{prefix}_Energy_Low"] = np.nan
        feats[f"{prefix}_Energy_Mid"] = np.nan
        feats[f"{prefix}_Energy_High"] = np.nan
        return feats

    total_power = np.sum(spectrum)

    # 1. Peak Frequency
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    feats[f"{prefix}_PeakFrequency"] = peak_freq

    # 2. Bandwidth (90% power)
    cum_power = np.cumsum(spectrum)
    power_90 = 0.90 * total_power
    start_idx = np.searchsorted(cum_power, (total_power - power_90))
    end_idx = np.searchsorted(cum_power, power_90)
    if start_idx >= len(freqs):
        start_idx = len(freqs) - 1
    if end_idx >= len(freqs):
        end_idx = len(freqs) - 1
    bandwidth = freqs[end_idx] - freqs[start_idx] if end_idx > start_idx else 0.0
    feats[f"{prefix}_Bandwidth"] = bandwidth

    # 3. Spectral Entropy
    p = spectrum / total_power
    spectral_entropy = -np.sum(p * np.log2(p + 1e-12))
    feats[f"{prefix}_SpectralEntropy"] = spectral_entropy

    # 4. Harmonic Energy Ratio
    harmonic_ratio = 0.0
    if peak_freq > 1e-6:
        tolerance = 0.02 * peak_freq
        max_harmonic = 5
        for k in range(max_harmonic):
            hf = peak_freq * (k + 1)
            idxs = np.where(np.abs(freqs - hf) <= tolerance)[0]
            harmonic_ratio += np.sum(spectrum[idxs])
        harmonic_ratio /= total_power
    feats[f"{prefix}_HarmonicEnergyRatio"] = harmonic_ratio

    # 5. Energy in Low/Mid/High frequency bands (example 0-3,3-10,>10)
    low_mask = (freqs >= 0) & (freqs < 3)
    mid_mask = (freqs >= 3) & (freqs < 10)
    high_mask = (freqs >= 10)
    feats[f"{prefix}_Energy_Low"] = np.sum(spectrum[low_mask])
    feats[f"{prefix}_Energy_Mid"] = np.sum(spectrum[mid_mask])
    feats[f"{prefix}_Energy_High"] = np.sum(spectrum[high_mask])

    return feats


def compute_cycle_features(signal, sampling_rate, prefix):
    """
    Cycle & zero-crossing features:
      - CycleCount
      - ZeroCrossingRate
      - CycleFrequency
      - CycleDuration
    """
    feats = {}
    n = len(signal)
    if n < 2:
        feats[f"{prefix}_CycleCount"] = np.nan
        feats[f"{prefix}_ZeroCrossingRate"] = np.nan
        feats[f"{prefix}_CycleFrequency"] = np.nan
        feats[f"{prefix}_CycleDuration"] = np.nan
        return feats

    signs = np.sign(signal)
    zc = np.sum(np.diff(signs) != 0)
    zero_cross_rate = zc / float(n)
    feats[f"{prefix}_ZeroCrossingRate"] = zero_cross_rate

    cycle_count = zc / 2.0
    feats[f"{prefix}_CycleCount"] = cycle_count

    total_time = n / float(sampling_rate)
    if total_time > 0:
        cycle_freq = cycle_count / total_time
    else:
        cycle_freq = np.nan
    feats[f"{prefix}_CycleFrequency"] = cycle_freq

    if cycle_freq > 1e-12:
        cycle_duration = 1.0 / cycle_freq
    else:
        cycle_duration = np.nan
    feats[f"{prefix}_CycleDuration"] = cycle_duration

    return feats


def compute_time_frequency_features(signal, sampling_rate, prefix):
    """
    Time-Frequency features:
      - InstantaneousFrequencyMean
      - FrequencyDrift
    """
    feats = {}
    n = len(signal)
    if n < 2:
        feats[f"{prefix}_InstantaneousFrequencyMean"] = np.nan
        feats[f"{prefix}_FrequencyDrift"] = np.nan
        return feats

    # short-time approach
    window_size = int(0.5 * sampling_rate)
    step_size = window_size // 2
    peak_freqs = []

    start = 0
    while start < n:
        end = min(start + window_size, n)
        segment = signal[start:end]
        freqs, spec = compute_frequency_spectrum(segment, sampling_rate)
        if len(freqs) > 1:
            idx_peak = np.argmax(spec)
            pf = freqs[idx_peak]
        else:
            pf = 0.0
        peak_freqs.append(pf)
        start += step_size

    if len(peak_freqs) == 0:
        feats[f"{prefix}_InstantaneousFrequencyMean"] = np.nan
        feats[f"{prefix}_FrequencyDrift"] = np.nan
        return feats

    inst_freq_mean = np.mean(peak_freqs)
    feats[f"{prefix}_InstantaneousFrequencyMean"] = inst_freq_mean

    times = np.arange(len(peak_freqs)) * (step_size / float(sampling_rate))
    if len(times) > 1:
        b = np.polyfit(times, peak_freqs, 1)
        freq_drift = b[0]
    else:
        freq_drift = 0.0
    feats[f"{prefix}_FrequencyDrift"] = freq_drift

    return feats


def compute_movement_frequency_features_for_repetition(arr, sampling_rate, prefix):
    """
    Master function that integrates the sub-feature calculations for a single 1D signal.
    """
    feats = {}
    freqs, spectrum = compute_frequency_spectrum(arr, sampling_rate)

    # 1) Statistical frequency distribution features
    feats.update(compute_statistical_freq_features(freqs, spectrum, prefix))

    # 2) Additional frequency-domain features
    feats.update(compute_frequency_domain_features(freqs, spectrum, prefix))

    # 3) Cycle & zero-crossing features
    feats.update(compute_cycle_features(arr, sampling_rate, prefix))

    # 4) Time-frequency features
    feats.update(compute_time_frequency_features(arr, sampling_rate, prefix))

    return feats


###############################################################################
# Main Extraction Pipeline
###############################################################################

def extract_frequency_features_from_file(csv_path, body_part, sensor_type, sampling_rate=100):
    """
    Reads a single CSV (X, Y, Z, Repetition), computes Movement Frequency features
    for each repetition, for each axis (X, Y, Z) + magnitude,
    embedding the sensor_type (acc or gyr) into the feature prefix.

    Returns a DataFrame with one row per (Subject, Repetition).
    """
    filename = os.path.basename(csv_path)
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {'X', 'Y', 'Z', 'Repetition'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing required columns: {required_cols - set(df.columns)}.")

    grouped = df.groupby('Repetition')
    all_rows = []
    for rep_id, group_data in grouped:
        gx = group_data['X'].to_numpy()
        gy = group_data['Y'].to_numpy()
        gz = group_data['Z'].to_numpy()
        g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

        # Build unique prefix: e.g., X_Shoulder_acc
        feats_x = compute_movement_frequency_features_for_repetition(
            gx, sampling_rate, prefix=f"X_{body_part}_{sensor_type}"
        )
        feats_y = compute_movement_frequency_features_for_repetition(
            gy, sampling_rate, prefix=f"Y_{body_part}_{sensor_type}"
        )
        feats_z = compute_movement_frequency_features_for_repetition(
            gz, sampling_rate, prefix=f"Z_{body_part}_{sensor_type}"
        )
        feats_mag = compute_movement_frequency_features_for_repetition(
            g_mag, sampling_rate, prefix=f"mag_{body_part}_{sensor_type}"
        )

        row_dict = {}
        row_dict.update(feats_x)
        row_dict.update(feats_y)
        row_dict.update(feats_z)
        row_dict.update(feats_mag)

        row_dict["Subject"] = subject_id
        row_dict["Repetition"] = rep_id

        all_rows.append(row_dict)

    return pd.DataFrame(all_rows)


def extract_all_movement_frequency_features(
        base_directory="processed_data_35_i",
        body_parts=('Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'),
        sampling_rate=100,
        output_csv="OutputCSVFiles/movement_frequency_features.csv",
        metadata_file=None
):
    """
    Iterates over each body_part and sensor folder ('acc' and 'gyr'),
    reads CSV files, computes Movement Frequency features,
    embedding sensor_type in the feature name (e.g., X_Shoulder_acc_MeanFrequency),
    and consolidates them into a single CSV.
    """
    all_dataframes = []

    for bp in body_parts:
        for sensor_type in ["acc", "gyr"]:
            sensor_folder = os.path.join(base_directory, bp, sensor_type)
            if not os.path.isdir(sensor_folder):
                print(f"[WARN] Folder does not exist: {sensor_folder}. Skipping.")
                continue

            csv_files = glob.glob(os.path.join(sensor_folder, "*.csv"))
            if not csv_files:
                print(f"[WARN] No CSV files in {sensor_folder}. Skipping.")
                continue

            for csv_path in csv_files:
                df_feats = extract_frequency_features_from_file(
                    csv_path, bp, sensor_type, sampling_rate
                )
                all_dataframes.append(df_feats)

    if not all_dataframes:
        print("[INFO] No frequency features extracted. Check data paths.")
        return

    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)
    # Group by (Subject, Repetition) to unify all columns
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    wide_df = wide_df.dropna(axis=1, how='all')
    # Drop columns where all values are empty strings
    wide_df = wide_df.drop(columns=wide_df.columns[wide_df.eq('').all()])

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Movement Frequency features saved to {output_csv}")
