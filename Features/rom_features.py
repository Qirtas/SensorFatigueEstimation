from numpy.polynomial.polynomial import polyfit
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch


def compute_additional_features_for_repetition(
        arr, prefix, body_part, sampling_rate=100
):
    """
    Computes new biomechanical + frequency domain features for a single 1D angular velocity signal:
      1) Biomechanical:
         - Peak Angular Velocity
         - Time to Peak Angular Velocity
         - Cumulative Angular Displacement (integral of velocity)
         - Angular Velocity Range (RoM) [Note: We had a simpler 'range' before, but re-state here if needed]
         - Symmetry Index
         - Smoothness Index (ratio of low-freq to high-freq power)
      2) Frequency Domain:
         - PSD (Power Spectral Density) [We can store summary stats or the full PSD]
         - Energy (sum of squared freq amplitudes)
         - Dominant Frequency
      3) Angle-Related (if we integrate velocity):
         - Max Angular Displacement
         - Min Angular Displacement
         - Range of displacement
         - Mean Angular Displacement
         - Angular Velocity RMS

    We return a dict of { feature_name : feature_value }.

    Parameters
    ----------
    arr : np.ndarray
        1D array of angular velocities for a single axis or magnitude.
    prefix : str
        Feature prefix (e.g. "X", "Y", "Z", or "mag").
    body_part : str
        Name of the body part (e.g. 'Shoulder').
    sampling_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    feats : dict
        Dictionary with keys for each new feature, with the axis and body_part embedded.
        Example: feats["Peak_AngVel_X_Shoulder"] = 3.52
    """

    feats = {}
    n = len(arr)
    if n < 1:
        # Handle edge case of empty repetition
        placeholders = [
            "Peak_AngVel", "TimeToPeak_AngVel", "CumulativeDisplacement",
            "AngVelRange", "SymmetryIndex", "SmoothnessIndex", "PSD_Energy",
            "DominantFreq", "MaxAngle", "MinAngle", "RangeAngle",
            "MeanAngle", "AngVel_RMS"
        ]
        for p in placeholders:
            feats[f"{p}_{prefix}_{body_part}"] = np.nan
        return feats

    dt = 1.0 / sampling_rate

    # ----------------------------------------------------
    # 1) Biomechanical Features
    # ----------------------------------------------------

    # A) Peak Angular Velocity (largest absolute velocity or just largest value?)


    peak_val = np.max(arr)
    feats[f"Peak_AngVel_{prefix}_{body_part}"] = peak_val

    # B) Time to Peak Angular Velocity
    idx_peak = np.argmax(arr)
    time_to_peak = idx_peak * dt
    feats[f"TimeToPeak_AngVel_{prefix}_{body_part}"] = time_to_peak

    # C) Cumulative Angular Displacement = ∫ angular velocity dt


    cumulative_disp = np.sum(arr) * dt
    feats[f"CumulativeDisplacement_{prefix}_{body_part}"] = cumulative_disp

    # D) Angular Velocity Range (RoM) = max(arr) - min(arr)
    angvel_range = np.max(arr) - np.min(arr)
    feats[f"AngVelRange_{prefix}_{body_part}"] = angvel_range

    # E) Symmetry Index
    #    Example: ratio comparing positive vs negative velocity.
    #    We'll define: SI = (|area_pos| - |area_neg|) / (|area_pos| + |area_neg|)
    #    area_pos = ∫ velocity dt where velocity>0, area_neg = ∫ velocity dt where velocity<0.
    pos_part = arr[arr > 0].sum() * dt
    neg_part = arr[arr < 0].sum() * dt  # this is negative
    si_denom = np.abs(pos_part) + np.abs(neg_part)
    if si_denom == 0:
        symmetry_index = 0.0
    else:
        symmetry_index = (np.abs(pos_part) - np.abs(neg_part)) / si_denom
    feats[f"SymmetryIndex_{prefix}_{body_part}"] = symmetry_index

    # F) Smoothness Index
    #    One approach: ratio of power in low freq (< cutoff) to power in high freq.
    #    We'll define a simple cutoff freq, e.g. 5 Hz.
    #    sum_of_power_in_low_freq / sum_of_power_in_high_freq
    cutoff_freq = 5.0  # example
    freqs_full, psd_full = welch(arr, fs=sampling_rate, nperseg=min(256, n))
    # split low vs high
    low_mask = freqs_full <= cutoff_freq
    high_mask = freqs_full > cutoff_freq
    power_low = np.sum(psd_full[low_mask])
    power_high = np.sum(psd_full[high_mask]) if np.sum(high_mask) > 0 else 1e-9
    smoothness_index = power_low / (power_high + 1e-9)  # avoid div zero
    feats[f"SmoothnessIndex_{prefix}_{body_part}"] = smoothness_index

    # ----------------------------------------------------
    # 2) Frequency Domain Features
    # ----------------------------------------------------


    total_energy = np.sum(psd_full)
    feats[f"PSD_Energy_{prefix}_{body_part}"] = total_energy

    # Dominant Frequency: freq bin with highest PSD
    idx_dom = np.argmax(psd_full)
    dominant_freq = freqs_full[idx_dom]
    feats[f"DominantFreq_{prefix}_{body_part}"] = dominant_freq

    # ----------------------------------------------------
    # 3) Angle-Related Features (integrating velocity -> angle)
    # ----------------------------------------------------


    angle_arr = np.cumsum(arr) * dt
    max_angle = np.max(angle_arr)
    min_angle = np.min(angle_arr)
    feats[f"MaxAngle_{prefix}_{body_part}"] = max_angle
    feats[f"MinAngle_{prefix}_{body_part}"] = min_angle
    feats[f"RangeAngle_{prefix}_{body_part}"] = max_angle - min_angle
    feats[f"MeanAngle_{prefix}_{body_part}"] = np.mean(angle_arr)

    # 4) Angular Velocity RMS
    #    sqrt(mean(velocity^2))
    angvel_rms = np.sqrt(np.mean(arr ** 2))
    feats[f"AngVel_RMS_{prefix}_{body_part}"] = angvel_rms

    return feats

def compute_rom_features_for_repetition(data, body_part="Shoulder", sampling_rate=100):


    gx = data['X'].to_numpy()
    gy = data['Y'].to_numpy()
    gz = data['Z'].to_numpy()

    # Compute magnitude
    g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # 1. Statistical Features (mean, median, min, max, range, std, var, skew, kurtosis)
    def describe(arr, prefix):
        stats = {}
        if len(arr) == 0:
            for metric in ['mean', 'median', 'min', 'max', 'range', 'std', 'var', 'skew', 'kurtosis']:
                stats[f"{prefix}_{metric}_{body_part}"] = np.nan
            return stats

        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min

        stats[f"{prefix}_mean_{body_part}"] = np.mean(arr)
        stats[f"{prefix}_median_{body_part}"] = np.median(arr)
        stats[f"{prefix}_min_{body_part}"] = arr_min
        stats[f"{prefix}_max_{body_part}"] = arr_max
        stats[f"{prefix}_range_{body_part}"] = arr_range
        stats[f"{prefix}_std_{body_part}"] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        stats[f"{prefix}_var_{body_part}"] = np.var(arr, ddof=1) if len(arr) > 1 else 0.0
        stats[f"{prefix}_skew_{body_part}"] = skew(arr) if len(arr) > 1 else 0.0
        stats[f"{prefix}_kurtosis_{body_part}"] = kurtosis(arr) if len(arr) > 1 else 0.0
        return stats

    feature_dict = {}
    feature_dict.update(describe(gx, prefix="ROM_X"))
    feature_dict.update(describe(gy, prefix="ROM_Y"))
    feature_dict.update(describe(gz, prefix="ROM_Z"))
    feature_dict.update(describe(g_mag, prefix="ROM_mag"))

    # 2. Temporal Features
    # A) Rate of Change (Slope of angular velocity)
    def compute_slope(arr, prefix):
        # Time array based on sampling rate
        t = np.arange(len(arr)) / sampling_rate
        if len(arr) < 2:
            return {f"{prefix}_slope_{body_part}": np.nan}
        # Fit a 1st-degree polynomial: arr ~ a + b*t. b is slope
        b, m = polyfit(t, arr, 1)  # polyfit returns [b, m] for a 1D poly (y = b + m*x)
        return {f"{prefix}_slope_{body_part}": m}

    feature_dict.update(compute_slope(gx, prefix="ROM_X"))
    feature_dict.update(compute_slope(gy, prefix="ROM_Y"))
    feature_dict.update(compute_slope(gz, prefix="ROM_Z"))
    feature_dict.update(compute_slope(g_mag, prefix="ROM_mag"))

    # B) Peak Frequency (FFT)
    #    We'll take the real FFT of the signal and find the frequency with the highest magnitude (dominant freq).
    def compute_peak_frequency(arr, prefix):
        if len(arr) < 2:
            return {f"{prefix}_peak_freq_{body_part}": np.nan}

        # rfft for real-valued signals
        fft_values = np.abs(rfft(arr))
        freqs = rfftfreq(len(arr), d=1.0 / sampling_rate)

        # Find index of maximum spectral amplitude (excluding the DC component at index 0 if you prefer)
        # If you want to exclude DC, start from index=1
        idx_peak = np.argmax(fft_values[1:]) + 1 if len(fft_values) > 1 else 0
        peak_freq = freqs[idx_peak]

        return {f"{prefix}_peak_freq_{body_part}": peak_freq}

    feature_dict.update(compute_peak_frequency(gx, prefix="ROM_X"))
    feature_dict.update(compute_peak_frequency(gy, prefix="ROM_Y"))
    feature_dict.update(compute_peak_frequency(gz, prefix="ROM_Z"))
    feature_dict.update(compute_peak_frequency(g_mag, prefix="ROM_mag"))

    # C) Number of Zero-Crossings
    #    We count sign changes from positive->negative or negative->positive.
    def compute_zero_crossings(arr, prefix):
        if len(arr) < 2:
            return {f"{prefix}_zero_crossings_{body_part}": np.nan}

        # Sign of each element
        signs = np.sign(arr)
        # Zero crossing if sign changes between consecutive samples
        zero_crosses = np.sum(np.diff(signs) != 0)

        return {f"{prefix}_zero_crossings_{body_part}": zero_crosses}

    feature_dict.update(compute_zero_crossings(gx, prefix="ROM_X"))
    feature_dict.update(compute_zero_crossings(gy, prefix="ROM_Y"))
    feature_dict.update(compute_zero_crossings(gz, prefix="ROM_Z"))
    feature_dict.update(compute_zero_crossings(g_mag, prefix="ROM_mag"))

    # D) Duration of Extreme Angles (or velocities)
    #    For simplicity, define "extreme" as being above some high percentile threshold
    #    (e.g., 90th percentile of absolute value). We sum the time spent above that threshold.
    def compute_duration_extreme(arr, prefix, percentile=90):
        if len(arr) < 1:
            return {f"{prefix}_extreme_duration_{body_part}": np.nan}

        # Let's do absolute velocities, so we capture both high + or - values
        abs_arr = np.abs(arr)
        # find threshold
        threshold = np.percentile(abs_arr, percentile)
        # count how many samples exceed threshold
        extreme_samples = np.sum(abs_arr >= threshold)
        # total time = extreme_samples / sampling_rate
        extreme_duration = extreme_samples / sampling_rate

        return {f"{prefix}_extreme_duration_{body_part}": extreme_duration}

    feature_dict.update(compute_duration_extreme(gx, prefix="ROM_X"))
    feature_dict.update(compute_duration_extreme(gy, prefix="ROM_Y"))
    feature_dict.update(compute_duration_extreme(gz, prefix="ROM_Z"))
    feature_dict.update(compute_duration_extreme(g_mag, prefix="ROM_mag"))

    feature_dict.update(compute_additional_features_for_repetition(
        gx, "X", body_part, sampling_rate
    ))

    feature_dict.update(compute_additional_features_for_repetition(
        gy, "Y", body_part, sampling_rate
    ))
    feature_dict.update(compute_additional_features_for_repetition(
        gz, "Z", body_part, sampling_rate
    ))
    feature_dict.update(compute_additional_features_for_repetition(
        g_mag, "mag", body_part, sampling_rate
    ))

    return feature_dict


def extract_rom_features_from_file(csv_path, body_part, sampling_rate=100):
    """
    Reads a single gyroscope CSV (columns: X, Y, Z, Repetition),
    computes both statistical + temporal RoM features for each repetition,
    returns a DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the segmented CSV file (gyroscope data).
    body_part : str
        The body part name (e.g., 'Shoulder').
    sampling_rate : int
        Gyroscope sampling frequency (default 100 Hz).

    Returns
    -------
    features_df : pandas.DataFrame
        One row per repetition with the computed RoM features.
    """
    filename = os.path.basename(csv_path)
    # Attempt to extract subject ID from filename (assuming "Subject_1_Upperarm_gyr.csv")
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    # Load CSV
    df = pd.read_csv(csv_path)

    required_cols = {'X', 'Y', 'Z', 'Repetition'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing required columns: {required_cols - set(df.columns)}.")

    # Group by Repetition
    grouped = df.groupby('Repetition')

    all_features = []
    for rep_id, group_data in grouped:
        # Compute combined (statistical + temporal) features
        feature_dict = compute_rom_features_for_repetition(group_data, body_part, sampling_rate)
        feature_dict['Subject'] = subject_id
        feature_dict['Repetition'] = rep_id
        all_features.append(feature_dict)

    features_df = pd.DataFrame(all_features)
    return features_df


def extract_all_rom_features(
        base_directory="processed_data_35_i",
        body_parts=('Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'),
        sampling_rate=100,
        output_csv="OutputCSVFiles/consolidated_rom_features.csv",
        metadata_file=None
):
    """
    Iterates over each body part's 'gyr' folder under base_directory,
    reads segmented gyroscope CSV files, computes both statistical and temporal RoM features,
    and consolidates them into one wide-format CSV.

    Parameters
    ----------
    base_directory : str
        Directory containing subfolders for each body part (e.g., base_directory/Pelvis/gyr/*.csv).
    body_parts : tuple or list of str
        Names of body parts to process.
    sampling_rate : int
        Gyroscope sampling frequency in Hz (default 100).
    output_csv : str
        Path to save the final consolidated CSV with all features.
    metadata_file : str or None
        Optional path to a CSV with subject-level data to merge on 'Subject'.
    """
    all_dfs = []

    # 1. Loop through body parts
    for bp in body_parts:
        gyr_folder = os.path.join(base_directory, bp, "gyr")
        if not os.path.isdir(gyr_folder):
            print(f"[WARN] Folder not found: {gyr_folder}. Skipping {bp}.")
            continue

        csv_files = glob.glob(os.path.join(gyr_folder, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {gyr_folder}. Skipping {bp}.")
            continue

        # 2. Compute features for each CSV
        for csv_path in csv_files:
            df_features = extract_rom_features_from_file(csv_path, bp, sampling_rate)
            all_dfs.append(df_features)

    if not all_dfs:
        print("[INFO] No RoM features extracted. Check data paths.")
        return

    # 3. Combine all dataframes
    combined_long_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 4. Pivot or group by (Subject, Repetition) to get wide format
    combined_long_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_long_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # 5. Optionally merge with metadata on 'Subject'
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # 6. Save to CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] RoM features (statistical + temporal) saved to {output_csv}")

# ----------------------------------------------------------------------------
# Example usage from main.py:
# if __name__ == "__main__":
#     extract_all_rom_features(
#         base_directory="processed_data_35_i",
#         body_parts=['Forearm','Pelvis','Palm','Shoulder','Torso','Upperarm'],
#         sampling_rate=100,
#         output_csv="OutputCSVFiles/consolidated_rom_features.csv",
#         metadata_file=None  # Or path to a metadata CSV if needed
#     )
