import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, skew


def compute_emg_features_for_repetition(
        emg_array, sampling_rate=1000, muscle_name="deltoideus_anterior"
):
    """
    Compute a set of IEMG (Integrated EMG) features for a single repetition of EMG data.

    Features:
      1) Total IEMG (sum of absolute EMG)
      2) IEMG per second = total IEMG / repetition_duration
      3) Normalized IEMG [0..1]: scale within this repetition
      4) Mean Absolute Value (MAV)
      5) Variance (of abs EMG)
      6) RMS (root-mean-square of EMG)
      7) Peak IEMG (max of abs EMG)
      8) Median and selected percentiles (25th, 75th)
      9) IEMG slope (linear fit slope of absolute EMG or cumulative IEMG)
      10) Cumulative IEMG (final value)
      11) Time-to-Peak IEMG

    Returns a dictionary { feature_name : value }.
    """
    feats = {}

    # Safety check
    if emg_array.size == 0:
        # Return NaNs for all features if empty
        placeholders = [
            "IEMG_Total", "IEMG_Per_Second", "IEMG_MAV", "IEMG_Variance", "IEMG_RMS",
            "IEMG_Peak", "IEMG_Median", "IEMG_25th", "IEMG_75th", "IEMG_Slope",
            "IEMG_Cumulative", "IEMG_TimeToPeak"
        ]
        for pl in placeholders:
            feats[f"{muscle_name}_{pl}"] = np.nan
        return feats

    # 1) Absolute EMG
    abs_emg = np.abs(emg_array)

    # Duration of repetition (in seconds)
    n_samples = len(emg_array)
    duration = n_samples / sampling_rate

    # 1) Total IEMG = sum of absolute EMG
    iemg_total = np.sum(abs_emg)
    feats[f"{muscle_name}_IEMG_Total"] = iemg_total

    # 2) IEMG per second
    iemg_per_sec = iemg_total / duration if duration > 0 else np.nan
    feats[f"{muscle_name}_IEMG_Per_Second"] = iemg_per_sec

    # 3) Normalized IEMG [0..1] within this repetition

    min_val = np.min(abs_emg)
    max_val = np.max(abs_emg)
    if max_val > min_val:
        norm_emg = (abs_emg - min_val) / (max_val - min_val)
    else:
        # edge case if all values are equal
        norm_emg = np.zeros_like(abs_emg)

    feats[f"{muscle_name}_IEMG_NormalizedMean"] = np.mean(norm_emg)

    # 4) Mean Absolute Value (MAV)
    mav = np.mean(abs_emg)
    feats[f"{muscle_name}_IEMG_MAV"] = mav

    # 5) Variance of abs EMG
    iemg_var = np.var(abs_emg, ddof=1) if abs_emg.size > 1 else 0.0
    feats[f"{muscle_name}_IEMG_Variance"] = iemg_var

    # 6) RMS
    rms_val = np.sqrt(np.mean(emg_array ** 2))
    feats[f"{muscle_name}_IEMG_RMS"] = rms_val

    # 7) Peak IEMG (max of abs EMG)
    peak_val = np.max(abs_emg)
    feats[f"{muscle_name}_IEMG_Peak"] = peak_val

    # 8) Median / Percentiles of EMG
    emg_median = np.median(abs_emg)
    feats[f"{muscle_name}_IEMG_Median"] = emg_median
    emg_25th = np.percentile(abs_emg, 25)
    emg_75th = np.percentile(abs_emg, 75)
    feats[f"{muscle_name}_IEMG_25th"] = emg_25th
    feats[f"{muscle_name}_IEMG_75th"] = emg_75th

    # 9) IEMG slope
    #    One approach: linear fit to cumulative sum of abs EMG vs time
    #    slope = d(IEMG)/dt. The final slope might reflect how quickly EMG accumulates.
    cumsum_emg = np.cumsum(abs_emg)
    if n_samples > 1:
        t = np.arange(n_samples) / sampling_rate
        # polynomial fit of cumsum_emg ~ a + b*t
        b = np.polyfit(t, cumsum_emg, 1)  # returns slope, intercept
        slope_val = b[0]  # slope
    else:
        slope_val = np.nan
    feats[f"{muscle_name}_IEMG_Slope"] = slope_val

    # 10) Cumulative IEMG (final value of cumsum)
    feats[f"{muscle_name}_IEMG_Cumulative"] = cumsum_emg[-1]

    # 11) Time-to-Peak IEMG (time at which abs_emg hits its max)
    idx_peak = np.argmax(abs_emg)
    time_to_peak = idx_peak / sampling_rate
    feats[f"{muscle_name}_IEMG_TimeToPeak"] = time_to_peak


    return feats


def extract_emg_features_from_file(csv_path, muscle_name, sampling_rate=1000):
    """
    Reads a CSV file with columns:
      EMG (mV), Repetition
    Groups by 'Repetition', then computes IEMG features for each repetition.

    Returns a DataFrame with columns:
      [Subject, Repetition, <muscle_name>_IEMG_Total, <muscle_name>_IEMG_RMS, ...]
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
        feats_dict = compute_emg_features_for_repetition(
            emg_array, sampling_rate=sampling_rate, muscle_name=muscle_name
        )

        # Add identifying info
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id

        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)


def extract_all_integrated_emg_features(
        base_directory="processed_data_35_i",
        muscle_folders=(
                "emg_deltoideus_anterior",
                "emg_deltoideus_posterior",
                # etc... add all muscle folder names here
        ),
        sampling_rate=1000,
        output_csv="OutputCSVFiles/consolidated_emg_features.csv",
        metadata_file=None
):
    """
    Iterates over each muscle folder (e.g., 'emg_deltoideus_anterior'),
    reads all CSV files (one per subject),
    computes IEMG features for each repetition,
    and consolidates results into a single CSV.

    Final CSV columns:
      Subject, Repetition, <MUSCLE>_IEMG_Total, <MUSCLE>_IEMG_RMS, ...

    Parameters
    ----------
    base_directory : str
        Root folder containing subfolders for each muscle.
    muscle_folders : list or tuple of str
        Subfolder names for each muscle.
    sampling_rate : float
        EMG sampling rate (default=1000 Hz).
    output_csv : str
        Path to save the final consolidated CSV.
    metadata_file : str or None
        Optional path to a CSV with additional subject-level data to merge on 'Subject'.
    """
    import os

    all_dataframes = []

    # 1. Loop through each muscle folder
    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Muscle folder not found: {muscle_path}. Skipping.")
            continue

        # 2. Find CSV files in this folder
        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        # 3. Process each CSV
        for csv_path in csv_files:
            print(f"[INFO] Processing {csv_path}...")
            df_emg_feats = extract_emg_features_from_file(
                csv_path, muscle_name=muscle_name, sampling_rate=sampling_rate
            )
            all_dataframes.append(df_emg_feats)

    if not all_dataframes:
        print("[INFO] No EMG features extracted. Check your directory structure and CSV files.")
        return

    # 4. Concatenate all results
    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # 5. Group by (Subject, Repetition) in case multiple muscles for the same repetition
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()

    # If you have multiple muscle data for the same Subject+Repetition, you might do .first() or pivot them.
    # Because we named columns with <muscle_name>_<feature>, the columns won't overlap.
    # If you want to merge multiple rows, you might do something else, but here .first() typically works.
    wide_df.reset_index(inplace=True)

    # 6. Optionally merge with metadata
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # 7. Save final CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Consolidated EMG features saved to {output_csv}")

