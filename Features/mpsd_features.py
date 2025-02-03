import os
import glob
import re
import numpy as np
import pandas as pd

from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis


def compute_psd(signal, sampling_rate):
    """
    Computes the Power Spectral Density (PSD) of a 1D signal using a real FFT.
    Returns:
      freqs: array of frequency bins (0 .. Nyquist)
      psd: power spectral density values for each bin
    """
    n = len(signal)
    if n < 2:
        # Edge case: not enough samples
        return np.array([0.0]), np.array([0.0])

    # Real FFT
    fft_vals = rfft(signal)
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    # Power spectral density (magnitude squared)
    psd = np.abs(fft_vals) ** 2
    return freqs, psd


def compute_mpsd_features(freqs, psd, prefix, low_freq_threshold=5.0):
    """
    Given frequency bins (freqs) and power spectral density (psd),
    compute MPSD-related features:
      1) Maximum Power Spectral Density (MPSD)
      2) Frequency of MPSD
      3) Power in Dominant Frequency Band (+/- 5% around freq_MPSD)
      4) Ratio of power in that band to total power
      5) Total Low-Frequency Power (below low_freq_threshold)
      6) Total High-Frequency Power (above low_freq_threshold)
      7) Low-to-High Frequency Power Ratio
      8) Total Power (Energy)
      9) PSD Skewness
      10) PSD Kurtosis
      11) Dominant Frequency Range (e.g., between 5th and 95th percentiles of PSD)

    prefix: label for the features (e.g., "X_Shoulder_acc", "mag_Forearm_gyr")
    low_freq_threshold: the frequency cut-off to separate low vs high freq power (default=5 Hz)
    """
    feats = {}
    n_bins = len(freqs)

    if n_bins < 2 or np.all(psd == 0):
        # Edge case: empty or no power
        placeholders = [
            "MPSD", "Freq_MPSD", "Power_Dominant_Band", "Ratio_Dominant_Band",
            "Total_LowFreq_Power", "Total_HighFreq_Power", "LowHigh_FreqRatio",
            "Total_Power", "PSD_Skewness", "PSD_Kurtosis", "DominantFreq_Range"
        ]
        for p in placeholders:
            feats[f"{prefix}_{p}"] = np.nan
        return feats

    # 1) Maximum Power Spectral Density (MPSD)
    idx_mpsd = np.argmax(psd)
    mpsd_val = psd[idx_mpsd]
    feats[f"{prefix}_MPSD"] = mpsd_val

    # 2) Frequency of MPSD
    freq_mpsd = freqs[idx_mpsd]
    feats[f"{prefix}_Freq_MPSD"] = freq_mpsd

    # 3) Power in Dominant Frequency Band (± 5% of freq_mpsd)
    #    If freq_mpsd == 0 or near zero, the "±5%" band might be 0. Adjust logic if needed.
    band_power = 0.0
    if freq_mpsd > 1e-12:
        band_low = freq_mpsd * 0.95
        band_high = freq_mpsd * 1.05
        band_mask = (freqs >= band_low) & (freqs <= band_high)
        band_power = np.sum(psd[band_mask])
    feats[f"{prefix}_Power_Dominant_Band"] = band_power

    # 4) Ratio of Power in Dominant Frequency Band
    total_power = np.sum(psd)
    ratio_dominant_band = band_power / total_power if total_power > 1e-12 else 0.0
    feats[f"{prefix}_Ratio_Dominant_Band"] = ratio_dominant_band

    # 5) Total Low-Frequency Power (< low_freq_threshold)
    low_mask = (freqs < low_freq_threshold)
    low_freq_power = np.sum(psd[low_mask])
    feats[f"{prefix}_Total_LowFreq_Power"] = low_freq_power

    # 6) Total High-Frequency Power (> low_freq_threshold)
    high_mask = (freqs >= low_freq_threshold)
    high_freq_power = np.sum(psd[high_mask])
    feats[f"{prefix}_Total_HighFreq_Power"] = high_freq_power

    # 7) Low-to-High Frequency Power Ratio
    #    Avoid divide-by-zero
    if high_freq_power > 1e-12:
        low_high_ratio = low_freq_power / high_freq_power
    else:
        low_high_ratio = np.inf
    feats[f"{prefix}_LowHigh_FreqRatio"] = low_high_ratio

    # 8) Total Power (Energy)
    feats[f"{prefix}_Total_Power"] = total_power

    # 9) PSD Skewness
    #    We treat psd as distribution across freq. Weighted skewness.
    #    Weighted mean freq = (sum(freq_i * psd_i) / total_power).
    #    Weighted stdev, etc. Then compute skew of PSD values themselves across bins or across freq?
    #    The user’s request says "skewness of the PSD values across frequencies,"
    #    so let's compute skewness on psd array itself (not freq distribution).
    #    So we do standard stats on the psd array.
    psd_mean = np.mean(psd)
    psd_std = np.std(psd, ddof=1) if len(psd) > 1 else 0.0
    if psd_std > 1e-12:
        psd_skew = np.mean(((psd - psd_mean) / psd_std) ** 3)
        psd_kurt = np.mean(((psd - psd_mean) / psd_std) ** 4) - 3.0
    else:
        psd_skew = 0.0
        psd_kurt = -3.0

    feats[f"{prefix}_PSD_Skewness"] = psd_skew

    # 10) PSD Kurtosis
    feats[f"{prefix}_PSD_Kurtosis"] = psd_kurt

    # 11) Dominant Frequency Range: freq range between e.g. 5th and 95th percentile of PSD cumsum
    cumsum_psd = np.cumsum(np.sort(psd))
    sorted_psd = np.sort(psd)
    total = cumsum_psd[-1]
    lower_thresh = 0.05 * total
    upper_thresh = 0.95 * total

    # find 5th percentile index
    lower_idx = np.searchsorted(cumsum_psd, lower_thresh)
    if lower_idx >= len(sorted_psd):
        lower_idx = len(sorted_psd) - 1
    # find 95th percentile index
    upper_idx = np.searchsorted(cumsum_psd, upper_thresh)
    if upper_idx >= len(sorted_psd):
        upper_idx = len(sorted_psd) - 1

    # We need to map sorted PSD back to frequencies. Because we sorted the PSD array alone,
    # we've lost direct mapping to freq bins. Another approach:
    #   - Sort by freq? Or
    #   - Or do a cumsum of PSD in freq order & find the freq bin for 5% & 95%.
    # We'll do the latter for consistency with "dominant freq range."
    psd_order = np.argsort(psd)  # indices that would sort PSD
    freq_sorted = freqs[psd_order]
    cumsum_psd_freq = np.cumsum(psd[psd_order])

    lower_idx = np.searchsorted(cumsum_psd_freq, lower_thresh)
    if lower_idx >= len(psd_order):
        lower_idx = len(psd_order) - 1

    upper_idx = np.searchsorted(cumsum_psd_freq, upper_thresh)
    if upper_idx >= len(psd_order):
        upper_idx = len(psd_order) - 1

    freq_range = abs(freq_sorted[upper_idx] - freq_sorted[lower_idx])
    feats[f"{prefix}_DominantFreq_Range"] = freq_range

    return feats


def compute_mpsd_for_repetition(arr, sampling_rate, prefix):
    """
    Wrapper to compute MPSD features for a single 1D signal (e.g., X-axis or magnitude).
    Returns a dictionary of the features, keyed by <prefix>_<FeatureName>.
    """
    freqs, psd = compute_psd(arr, sampling_rate)
    feats = compute_mpsd_features(freqs, psd, prefix)
    return feats


def extract_mpsd_features_from_file(csv_path, body_part, sensor_type, sampling_rate=100):
    """
    Reads a single CSV (with columns: X, Y, Z, Repetition), then for each repetition,
    computes MPSD features for X, Y, Z, and magnitude, embedding sensor_type + body part
    in the feature names (e.g., MPSD_X_Shoulder_acc, etc.).

    Returns a DataFrame with one row per (Subject, Repetition).
    """
    filename = os.path.basename(csv_path)
    # Infer subject ID
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"X", "Y", "Z", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing {required_cols - set(df.columns)}.")

    grouped = df.groupby("Repetition")
    all_rows = []
    for rep_id, group_data in grouped:
        gx = group_data["X"].to_numpy()
        gy = group_data["Y"].to_numpy()
        gz = group_data["Z"].to_numpy()
        g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

        # Prepare prefixes, e.g. "X_Shoulder_acc"
        pref_x = f"X_{body_part}_{sensor_type}"
        pref_y = f"Y_{body_part}_{sensor_type}"
        pref_z = f"Z_{body_part}_{sensor_type}"
        pref_mag = f"mag_{body_part}_{sensor_type}"

        feats_x = compute_mpsd_for_repetition(gx, sampling_rate, prefix=pref_x)
        feats_y = compute_mpsd_for_repetition(gy, sampling_rate, prefix=pref_y)
        feats_z = compute_mpsd_for_repetition(gz, sampling_rate, prefix=pref_z)
        feats_mag = compute_mpsd_for_repetition(g_mag, sampling_rate, prefix=pref_mag)

        row_dict = {}
        row_dict.update(feats_x)
        row_dict.update(feats_y)
        row_dict.update(feats_z)
        row_dict.update(feats_mag)

        row_dict["Subject"] = subject_id
        row_dict["Repetition"] = rep_id

        all_rows.append(row_dict)

    return pd.DataFrame(all_rows)


def extract_all_mpsd_features(
        base_directory="processed_data_35_i",
        body_parts=('Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'),
        sampling_rate=100,
        output_csv="OutputCSVFiles/mpsd_features.csv",
        metadata_file=None
):
    """
    Iterates over each body_part and sensor folder ('acc', 'gyr'),
    reads CSV files (X, Y, Z, Repetition),
    computes MPSD-related features for each axis + magnitude,
    and consolidates them into a single CSV with one row per repetition per subject.

    The final dataset columns look like:
      MPSD_X_Shoulder_acc, Freq_MPSD_X_Shoulder_acc, ..., Subject, Repetition, ...

    Parameters
    ----------
    base_directory : str
        The root directory containing subfolders for each body part, then 'acc'/'gyr'.
    body_parts : list or tuple of str
        The body parts to process (default set shown).
    sampling_rate : int
        IMU sampling frequency (default=100 Hz).
    output_csv : str
        Path to save the final consolidated CSV (default="OutputCSVFiles/mpsd_features.csv").
    metadata_file : str or None
        Optional path to a CSV with subject-level info. Merged on "Subject".
    """
    all_dfs = []

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
                df_feats = extract_mpsd_features_from_file(
                    csv_path, bp, sensor_type, sampling_rate
                )
                all_dfs.append(df_feats)

    if not all_dfs:
        print("[INFO] No MPSD features extracted. Check your data paths.")
        return

    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Group by (Subject, Repetition) to unify columns
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
    print(f"[INFO] MPSD features saved to {output_csv}")
