import os
import glob
import re
import numpy as np
import pandas as pd

def compute_zc_features_for_repetition(emg_array, sampling_rate=1000, muscle_name="deltoideus_anterior"):
    """
    Compute zero crossing based features for one repetition of EMG data.

    Parameters:
      emg_array : numpy array
          Array of EMG samples for one repetition.
      sampling_rate : int or float
          Sampling rate of the EMG signal.
      muscle_name : str
          Muscle name (used as a prefix for feature names).

    Returns:
      feats : dict
          Dictionary containing the following features:
            - <muscle>_ZC_Total       : Total zero crossings.
            - <muscle>_ZC_Rate        : Zero crossing rate (per second).
            - <muscle>_ZC_MeanInterval: Mean time interval between crossings.
            - <muscle>_ZC_StdInterval : Standard deviation of the intervals.
            - <muscle>_ZC_MinInterval : Minimum interval between crossings.
            - <muscle>_ZC_MaxInterval : Maximum interval between crossings.
    """
    feats = {}

    # Safety check: if signal is too short, return NaNs.
    if emg_array.size < 2:
        placeholders = [
            "ZC_Total", "ZC_Rate", "ZC_MeanInterval",
            "ZC_StdInterval", "ZC_MinInterval", "ZC_MaxInterval"
        ]
        for ph in placeholders:
            feats[f"{muscle_name}_{ph}"] = np.nan
        return feats

    # Compute zero crossings:
    # A zero crossing occurs when the sign of the signal changes.
    zc_indices = np.where(np.diff(np.sign(emg_array)) != 0)[0]
    zc_total = zc_indices.size
    feats[f"{muscle_name}_ZC_Total"] = zc_total

    # Duration of the repetition in seconds.
    duration_sec = emg_array.size / sampling_rate

    # Zero crossing rate (number per second)
    zc_rate = zc_total / duration_sec if duration_sec > 0 else np.nan
    feats[f"{muscle_name}_ZC_Rate"] = zc_rate

    # Compute intervals between zero crossings (in seconds)
    if zc_total > 1:
        intervals = np.diff(zc_indices) / sampling_rate
        feats[f"{muscle_name}_ZC_MeanInterval"] = np.mean(intervals)
        feats[f"{muscle_name}_ZC_StdInterval"] = np.std(intervals, ddof=1) if intervals.size > 1 else 0.0
        feats[f"{muscle_name}_ZC_MinInterval"] = np.min(intervals)
        feats[f"{muscle_name}_ZC_MaxInterval"] = np.max(intervals)
    else:
        # Not enough crossings to compute intervals.
        feats[f"{muscle_name}_ZC_MeanInterval"] = np.nan
        feats[f"{muscle_name}_ZC_StdInterval"] = np.nan
        feats[f"{muscle_name}_ZC_MinInterval"] = np.nan
        feats[f"{muscle_name}_ZC_MaxInterval"] = np.nan

    return feats

def extract_emg_zc_features_from_file(csv_path, muscle_name, sampling_rate=1000):
    """
    Reads a CSV file with EMG data (columns: EMG, Repetition),
    groups by repetition, and computes zero crossing features per repetition.

    Returns a DataFrame with columns:
        [Subject, Repetition, <muscle>_ZC_Total, <muscle>_ZC_Rate, ...]
    """
    filename = os.path.basename(csv_path)
    # Extract subject ID from the filename using a regular expression.
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"EMG", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} is missing required columns: {required_cols - set(df.columns)}")

    grouped = df.groupby("Repetition")
    all_feature_rows = []
    for rep_id, group_data in grouped:
        emg_array = group_data["EMG"].to_numpy()
        feats_dict = compute_zc_features_for_repetition(emg_array, sampling_rate=sampling_rate, muscle_name=muscle_name)
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id
        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)

def extract_all_emg_zc_features(
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
    output_csv="Features/Extracted/zeroCrossing_features_EMG.csv",
    metadata_file=None
):
    """
    Iterates over each muscle folder in the specified base directory,
    reads all EMG CSV files, computes zero crossing features for each repetition,
    and consolidates the results into a single CSV file.

    The output CSV will have the following columns:
        Subject, Repetition, <MUSCLE>_ZC_Total, <MUSCLE>_ZC_Rate,
        <MUSCLE>_ZC_MeanInterval, <MUSCLE>_ZC_StdInterval,
        <MUSCLE>_ZC_MinInterval, <MUSCLE>_ZC_MaxInterval, ...

    If a metadata CSV file is provided, it will be merged on the 'Subject' column.
    """
    all_dataframes = []
    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Muscle folder not found: {muscle_path}. Skipping.")
            continue

        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        for csv_path in csv_files:
            print(f"[INFO] Processing {csv_path} ...")
            df_emg_feats = extract_emg_zc_features_from_file(csv_path, muscle_name=muscle_name, sampling_rate=sampling_rate)
            all_dataframes.append(df_emg_feats)

    if not all_dataframes:
        print("[INFO] No zero crossing features were extracted. Please verify your data directories and file formats.")
        return

    # Combine all dataframes so that each row corresponds to a unique (Subject, Repetition)
    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # Optionally merge with metadata if provided.
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # Ensure that the output directory exists.
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Consolidated zero crossing features saved to {output_csv}")

if __name__ == "__main__":
    # Example call from the main file:
    extract_all_emg_zc_features(
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
        output_csv="Features/Extracted/zeroCrossing_features_EMG.csv",
        metadata_file=None  # Replace with path to metadata CSV if available
    )