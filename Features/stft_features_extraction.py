import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import stft, welch
from scipy.stats import entropy
from scipy.stats import skew


def compute_stft_features_for_repetition(
        emg_array, sampling_rate=1000, muscle_name="deltoideus_anterior",
        window_size=256, overlap=128
):
    """
    Compute a set of STFT-based features for a single repetition of EMG data.
    Parameters:
        emg_array (np.ndarray): Raw EMG signal.
        sampling_rate (int): Sampling rate of the EMG signal (default=1000 Hz).
        muscle_name (str): Name of the muscle for naming features.
        window_size (int): Size of the STFT window (default=256 samples).
        overlap (int): Overlap between consecutive windows (default=128 samples).
    Returns:
        dict: Dictionary of STFT-based features.
    """
    feats = {}

    # Safety check
    if emg_array.size == 0:
        placeholders = [
            "STFT_MNF", "STFT_MDF", "STFT_PeakFreq", "STFT_TotalPower",
            "STFT_SpectralEntropy", "STFT_LowFreqPower", "STFT_HighFreqPower",
            "STFT_FreqBandRatio", "STFT_SpectralSlope", "STFT_MNF_Std",
            "STFT_MDF_Std", "STFT_Skewness"
        ]
        for pl in placeholders:
            feats[f"{muscle_name}_{pl}"] = np.nan
        return feats

    # Perform STFT
    f, t, Zxx = stft(emg_array, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
    power_spectrum = np.abs(Zxx) ** 2  # Power spectrum
    mean_power_spectrum = np.mean(power_spectrum, axis=1)  # Average over time windows

    # 1) Mean Frequency (MNF)
    mnf = np.sum(f * mean_power_spectrum) / np.sum(mean_power_spectrum)
    feats[f"{muscle_name}_STFT_MNF"] = mnf

    # 2) Median Frequency (MDF)
    cumulative_power = np.cumsum(mean_power_spectrum)
    mdf_index = np.argmin(np.abs(cumulative_power - 0.5 * cumulative_power[-1]))
    mdf = f[mdf_index]
    feats[f"{muscle_name}_STFT_MDF"] = mdf

    # 3) Peak Frequency
    peak_freq_index = np.argmax(mean_power_spectrum)
    peak_freq = f[peak_freq_index]
    feats[f"{muscle_name}_STFT_PeakFreq"] = peak_freq

    # 4) Total Power
    total_power = np.sum(mean_power_spectrum)
    feats[f"{muscle_name}_STFT_TotalPower"] = total_power

    # 5) Spectral Entropy
    normalized_power = mean_power_spectrum / np.sum(mean_power_spectrum)
    spectral_entropy = entropy(normalized_power, base=2)
    feats[f"{muscle_name}_STFT_SpectralEntropy"] = spectral_entropy

    # 6) Frequency Band Power Ratios
    low_freq_mask = (f >= 0) & (f <= 50)
    high_freq_mask = (f > 50) & (f <= 500)
    low_freq_power = np.sum(mean_power_spectrum[low_freq_mask])
    high_freq_power = np.sum(mean_power_spectrum[high_freq_mask])
    freq_band_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else np.nan
    feats[f"{muscle_name}_STFT_LowFreqPower"] = low_freq_power
    feats[f"{muscle_name}_STFT_HighFreqPower"] = high_freq_power
    feats[f"{muscle_name}_STFT_FreqBandRatio"] = freq_band_ratio

    # 7) Spectral Slope
    coeffs = np.polyfit(f, mean_power_spectrum, 1)
    spectral_slope = coeffs[0]  # Slope of the linear fit
    feats[f"{muscle_name}_STFT_SpectralSlope"] = spectral_slope

    # 8) Standard Deviation of MNF/MDF
    mnf_std = np.std([np.sum(f * ps) / np.sum(ps) for ps in power_spectrum.T])
    mdf_std = np.std([f[np.argmin(np.abs(np.cumsum(ps) - 0.5 * np.sum(ps)))] for ps in power_spectrum.T])
    feats[f"{muscle_name}_STFT_MNF_Std"] = mnf_std
    feats[f"{muscle_name}_STFT_MDF_Std"] = mdf_std

    # 9) Skewness of the Spectrum
    skewness = skew(mean_power_spectrum)
    feats[f"{muscle_name}_STFT_Skewness"] = skewness

    return feats


def extract_stft_features_from_file(csv_path, muscle_name, sampling_rate=1000):
    """
    Reads a CSV file with columns:
      EMG (mV), Repetition
    Groups by 'Repetition', then computes STFT features for each repetition.
    Returns a DataFrame with columns:
      [Subject, Repetition, <muscle_name>_STFT_MNF, <muscle_name>_STFT_MDF, ...]
    """
    filename = os.path.basename(csv_path)
    # Attempt to parse subject ID from filename (e.g., "Subject_1_emg_deltoideus_anterior.csv")
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    if subject_match:
        subject_id = subject_match.group(1)
    else:
        subject_id = "Unknown"

    # Load CSV
    df = pd.read_csv(csv_path)
    required_cols = {"EMG", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing columns {required_cols - set(df.columns)}.")

    # Group by Repetition
    grouped = df.groupby("Repetition")
    all_feature_rows = []
    for rep_id, group_data in grouped:
        emg_array = group_data["EMG"].to_numpy()
        # Compute features
        feats_dict = compute_stft_features_for_repetition(
            emg_array, sampling_rate=sampling_rate, muscle_name=muscle_name
        )
        # Add identifying info
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id
        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)


def extract_all_stft_features(
        base_directory="processed_data_35_i",
        muscle_folders=[
            "emg_deltoideus_anterior",
            "emg_deltoideus_posterior",
            "emg_infraspinatus",
            "emg_latissimus_dorsi",
            "emg_pectoralis_major",
            "emg_trapezius_ascendens"
        ],
        sampling_rate=1000,
        output_csv="Features/Extracted/stft_features_EMG.csv",
        metadata_file=None
):
    """
    Iterates over each muscle folder, reads all CSV files, computes STFT features for each repetition,
    and consolidates results into a single CSV.
    Final CSV columns:
      Subject, Repetition, <MUSCLE>_STFT_MNF, <MUSCLE>_STFT_MDF, ...
    """
    import os
    all_dataframes = []

    # Loop through each muscle folder
    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Muscle folder not found: {muscle_path}. Skipping.")
            continue

        # Find CSV files in this folder
        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        # Process each CSV
        for csv_path in csv_files:
            print(f"[INFO] Processing {csv_path}...")
            df_stft_feats = extract_stft_features_from_file(
                csv_path, muscle_name=muscle_name, sampling_rate=sampling_rate
            )
            all_dataframes.append(df_stft_feats)

    if not all_dataframes:
        print("[INFO] No STFT features extracted. Check your directory structure and CSV files.")
        return

    # Concatenate all results
    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # Group by (Subject, Repetition)
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # Optionally merge with metadata
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # Save final CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Consolidated STFT features saved to {output_csv}")

