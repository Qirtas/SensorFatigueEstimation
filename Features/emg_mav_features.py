import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


###############################################################################
# Helper Functions
###############################################################################

def compute_mav_features_for_repetition(
        emg_array,
        sampling_rate=1000,
        muscle_name="deltoideus_anterior"
):
    """
    Computes MAV-based features from the raw EMG data (in mV) for a single repetition.
    The raw data is segmented for this repetition. We'll consider the time series:
      M(t) = abs( EMG(t) ).

    Features included:
      1) Basic MAV
         - Total MAV (sum of abs EMG)
         - MAV Per Second
      2) Normalized MAV (2 ways)
         - Min-Max Normalized (0..1)
         - Z-Score Normalized
      3) MAV Slope
         - Linear slope of abs(EMG) over time
      4) MAV Variability
         - SD of abs(EMG)
         - CV = SD / mean(M)
      5) MAV Percentiles
         - Median MAV (50th percentile)
         - Percentile Range MAV (Q3 - Q1)
      6) MAV Peaks
         - Maximum MAV
         - Time to Peak MAV
      7) Temporal MAV Patterns
         - Cumulative MAV (final cumsum)
         - MAV Rise Time (time from baseline to peak)
         - MAV Decay Time (time from peak back to baseline)
      8) Fatigue-Related MAV Features (placeholders)
         - MAV Fatigue Index (decline rate within the repetition or across repetitions)
         - MAV Plateau Index (whether MAV stabilizes near the end)

    Returns
    -------
    feats : dict
        Dictionary keyed by <muscle_name>_<FeatureName> => value
    """
    feats = {}

    # If no data in this repetition, return NaNs
    if len(emg_array) == 0:
        placeholders = [
            "MAV_Total", "MAV_Per_Sec", "MAV_MinMaxNorm", "MAV_ZScoreNorm",
            "MAV_Slope", "MAV_SD", "MAV_CV", "MAV_Median", "MAV_IQR",
            "MAV_Max", "MAV_TimeToPeak", "MAV_Cumulative", "MAV_RiseTime",
            "MAV_DecayTime"
        ]
        for p in placeholders:
            feats[f"{muscle_name}_{p}"] = np.nan
        return feats

    # 1. Basic MAV
    abs_emg = np.abs(emg_array)
    n_samples = len(emg_array)
    duration = n_samples / sampling_rate

    # (a) Total MAV = sum(abs(EMG))
    mav_total = np.sum(abs_emg)
    feats[f"{muscle_name}_MAV_Total"] = mav_total

    # (b) MAV per second = MAV_Total / repetition duration
    mav_per_sec = mav_total / duration if duration > 0 else np.nan
    feats[f"{muscle_name}_MAV_Per_Sec"] = mav_per_sec

    # 2. Normalized MAV
    # (a) Min-Max Normalized (0..1) across this repetition
    min_val = np.min(abs_emg)
    max_val = np.max(abs_emg)
    if max_val > min_val:
        norm_minmax = (abs_emg - min_val) / (max_val - min_val)
    else:
        # all values might be identical
        norm_minmax = np.zeros_like(abs_emg)
    # For a single feature, let's store the mean of the min-max normalized signal
    feats[f"{muscle_name}_MAV_MinMaxNorm"] = np.mean(norm_minmax)

    # (b) Z-score Normalized => (abs_emg - mean) / std
    mean_val = np.mean(abs_emg)
    std_val = np.std(abs_emg, ddof=1) if n_samples > 1 else 0.0
    if std_val > 1e-12:
        zscore_vals = (abs_emg - mean_val) / std_val
    else:
        zscore_vals = np.zeros_like(abs_emg)
    # We'll store mean of the z-score distribution (which should be near 0),
    # or if you prefer, the max or RMS. We'll do mean here for demonstration.
    feats[f"{muscle_name}_MAV_ZScoreNorm"] = np.mean(zscore_vals)

    # 3. MAV Slope => linear slope of abs_emg(t) over time
    # We can do a polyfit with t vs. abs_emg
    t = np.arange(n_samples) / sampling_rate
    if n_samples > 1:
        slope, intercept = np.polyfit(t, abs_emg, 1)  # slope is slope
        feats[f"{muscle_name}_MAV_Slope"] = slope
    else:
        feats[f"{muscle_name}_MAV_Slope"] = np.nan

    # 4. MAV Variability
    # (a) Standard Deviation
    feats[f"{muscle_name}_MAV_SD"] = std_val
    # (b) Coefficient of Variation (CV = SD / mean(MAV))
    if mean_val > 1e-12:
        cv_val = std_val / mean_val
    else:
        cv_val = np.nan
    feats[f"{muscle_name}_MAV_CV"] = cv_val

    # 5. MAV Percentiles
    mav_median = np.median(abs_emg)
    feats[f"{muscle_name}_MAV_Median"] = mav_median
    q1 = np.percentile(abs_emg, 25)
    q3 = np.percentile(abs_emg, 75)
    feats[f"{muscle_name}_MAV_IQR"] = q3 - q1

    # 6. MAV Peaks
    # (a) Maximum MAV (peak)
    mav_max = max_val
    feats[f"{muscle_name}_MAV_Max"] = mav_max
    # (b) Time to Peak MAV
    idx_peak = np.argmax(abs_emg)
    time_to_peak = idx_peak / sampling_rate
    feats[f"{muscle_name}_MAV_TimeToPeak"] = time_to_peak

    # 7. Temporal MAV Patterns
    # (a) Cumulative MAV => final sum is the same as mav_total, but we can store it
    feats[f"{muscle_name}_MAV_Cumulative"] = mav_total

    # (b) MAV Rise Time => how long does it take to go from "baseline" to peak?
    # We'll define baseline as e.g. 10% of peak. Then find the time index
    # from the start until abs_emg first exceeds 10% * mav_max
    # If mav_max = 0, skip
    if mav_max > 1e-12:
        threshold_rise = 0.1 * mav_max
        rise_idx = np.where(abs_emg >= threshold_rise)[0]
        if len(rise_idx) > 0:
            mav_rise_time = (rise_idx[0]) / sampling_rate
        else:
            mav_rise_time = np.nan
    else:
        mav_rise_time = np.nan
    feats[f"{muscle_name}_MAV_RiseTime"] = mav_rise_time

    # (c) MAV Decay Time => from peak back to baseline (10% of peak?)
    # We'll define after idx_peak, how long until it goes below threshold again
    if mav_max > 1e-12:
        post_peak = abs_emg[idx_peak:]
        decay_idx = np.where(post_peak <= threshold_rise)[0]
        if len(decay_idx) > 0:
            # index is relative to idx_peak
            mav_decay_time = (idx_peak + decay_idx[0]) / sampling_rate - time_to_peak
        else:
            mav_decay_time = np.nan
    else:
        mav_decay_time = np.nan
    feats[f"{muscle_name}_MAV_DecayTime"] = mav_decay_time

    # # 8. Fatigue-Related MAV Features (placeholders)
    # # Typically, you'd compare across multiple repetitions or time windows
    # # For this single repetition, we can define:
    # # (a) MAV_FatigueIndex => e.g., slope from start to end of repetition
    # # Already we have MAV_Slope of abs_emg, so let's just reuse that as a placeholder
    # # Or you might measure the decline from first half to second half. Let's do that:
    # half_idx = n_samples // 2
    # first_half_mean = np.mean(abs_emg[:half_idx]) if half_idx > 0 else mean_val
    # second_half_mean = np.mean(abs_emg[half_idx:]) if half_idx < n_samples else mean_val
    # if first_half_mean > 1e-12:
    #     fatigue_index = (second_half_mean - first_half_mean) / first_half_mean
    # else:
    #     fatigue_index = np.nan
    # feats[f"{muscle_name}_MAV_FatigueIndex"] = fatigue_index
    #
    # # (b) MAV_PlateauIndex => checks if last quarter vs. third quarter are stable
    # quarter_idx = n_samples // 4
    # third_quarter = abs_emg[2 * quarter_idx:3 * quarter_idx] if 3 * quarter_idx <= n_samples else []
    # last_quarter = abs_emg[3 * quarter_idx:] if 3 * quarter_idx <= n_samples else []
    # if len(third_quarter) > 0 and len(last_quarter) > 0:
    #     third_mean = np.mean(third_quarter)
    #     last_mean = np.mean(last_quarter)
    #     # if difference is small, it might "plateau"
    #     plateau_index = last_mean - third_mean
    # else:
    #     plateau_index = np.nan
    # feats[f"{muscle_name}_MAV_PlateauIndex"] = plateau_index

    return feats


def extract_mav_features_from_file(csv_path, muscle_name, sampling_rate=1000):
    """
    Reads one CSV (EMG, Repetition), groups by Repetition,
    and computes MAV-based features for each repetition.

    Output columns:
      Subject, Repetition, <muscle>_MAV_...
    """
    filename = os.path.basename(csv_path)
    # Attempt to parse subject ID
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"EMG", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing {required_cols - set(df.columns)}.")

    grouped = df.groupby("Repetition")
    all_rows = []
    for rep_id, group_data in grouped:
        emg_array = group_data["EMG"].values
        feats = compute_mav_features_for_repetition(
            emg_array, sampling_rate, muscle_name
        )
        feats["Subject"] = subject_id
        feats["Repetition"] = rep_id
        all_rows.append(feats)

    return pd.DataFrame(all_rows)


def extract_all_mav_features(
        base_directory="processed_data_35_i",
        muscle_folders=(
                "emg_deltoideus_anterior",
                "emg_deltoideus_posterior",
                # other muscle folders
        ),
        sampling_rate=1000,
        output_csv="OutputCSVFiles/consolidated_mav_features.csv",
        metadata_file=None
):
    """
    Iterates over each muscle folder, reads segmented EMG CSVs,
    computes MAV-based features for each repetition, merges all
    into a single wide CSV:
      Subject, Repetition, <muscle>_MAV_Total, <muscle>_MAV_Slope, ...
    """
    import os

    all_dfs = []

    # 1. Loop through each muscle folder
    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Folder not found: {muscle_path}. Skipping {muscle_name}.")
            continue

        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        for csv_path in csv_files:
            print(f"[INFO] Processing EMG file: {csv_path}")
            df_mav_feats = extract_mav_features_from_file(
                csv_path, muscle_name, sampling_rate
            )
            all_dfs.append(df_mav_feats)

    if not all_dfs:
        print("[INFO] No MAV features extracted. Check your directories and CSV files.")
        return

    # 2. Combine all data
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 3. Group by (Subject, Repetition) and combine columns
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # 4. Optionally merge with metadata on 'Subject'
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # 5. Save final CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] MAV-based EMG features saved to {output_csv}")
