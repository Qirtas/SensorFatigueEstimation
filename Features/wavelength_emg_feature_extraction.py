# wavelength_emg_feature_extraction.py
import os
import glob
import re
import numpy as np
import pandas as pd

def compute_emg_wavelength_features_for_repetition(
    emg_array, sampling_rate=1000, muscle_name="deltoideus_anterior"
):
    """
    Compute Wavelength (WV) form-based features for a single EMG repetition.

    Features included:
      1) WV_Total        : sum of absolute sample-to-sample differences
      2) WV_Per_Second   : WV_Total / repetition_duration
      3) WV_Slope        : linear slope of the cumulative wave length over time
      4) TimeToHalfWV    : time to reach 50% of total waveform length
      5) WV_Variance     : variance of the sample-to-sample differences

    Returns:
      Dictionary of { <muscle_name>_<feature_name>: value }
    """

    feats = {}

    # Handle empty arrays
    if emg_array.size == 0:
        placeholders = [
            "WV_Total",
            "WV_Per_Second",
            "WV_Slope",
            "TimeToHalfWV",
            "WV_Variance"
        ]
        for pl in placeholders:
            feats[f"{muscle_name}_{pl}"] = np.nan
        return feats

    # Duration of the repetition (in seconds)
    n_samples = len(emg_array)
    duration = n_samples / sampling_rate

    # 1) Compute Waveform Length (WV_Total)
    #    WV = sum of absolute differences
    abs_diff = np.abs(np.diff(emg_array))
    wv_total = np.sum(abs_diff)
    feats[f"{muscle_name}_WV_Total"] = wv_total

    # 2) WV per second (WV_Total / repetition_duration)
    wv_per_second = wv_total / duration if duration > 0 else np.nan
    feats[f"{muscle_name}_WV_Per_Second"] = wv_per_second

    # 3) WV_Slope
    #    We'll consider the cumulative sum of "abs_diff" over time and fit a line
    #    to see how quickly the wave length accumulates.
    cumsum_wv = np.cumsum(abs_diff)
    if n_samples > 1:
        # We'll create a time vector with n_samples - 1 points (since we used diff)
        t = np.arange(len(abs_diff)) / sampling_rate
        slope, intercept = np.polyfit(t, cumsum_wv, 1)
    else:
        slope = np.nan
    feats[f"{muscle_name}_WV_Slope"] = slope

    # 4) TimeToHalfWV
    #    The time at which the cumulative wave length is 50% of wv_total
    if wv_total > 0:
        half_wv = wv_total / 2.0
        idx_half = np.searchsorted(cumsum_wv, half_wv)
        if idx_half < len(cumsum_wv):
            time_to_half_wv = idx_half / sampling_rate
        else:
            time_to_half_wv = duration
    else:
        time_to_half_wv = np.nan
    feats[f"{muscle_name}_TimeToHalfWV"] = time_to_half_wv

    # 5) WV_Variance
    #    Variance in abs_diff can also serve as a measure of EMG complexity changes.
    if len(abs_diff) > 1:
        wv_variance = np.var(abs_diff, ddof=1)  # sample variance
    else:
        wv_variance = 0.0
    feats[f"{muscle_name}_WV_Variance"] = wv_variance

    return feats


def extract_emg_wavelength_features_from_file(csv_path, muscle_name, sampling_rate=1000):
    """
    Reads a CSV file with columns: [EMG, Repetition]
    Groups by 'Repetition' and computes Wavelength-based EMG features for each repetition.

    Returns a DataFrame with:
      [Subject, Repetition, <muscle_name>_WV_Total, <muscle_name>_WV_Per_Second, etc...]
    """
    import pandas as pd
    filename = os.path.basename(csv_path)

    # Attempt to parse subject ID from filename (e.g. "Subject_34_emg_deltoideus_anterior.csv")
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    if subject_match:
        subject_id = subject_match.group(1)
    else:
        subject_id = "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"EMG", "Repetition"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"File {csv_path} is missing columns: {missing}")

    # Group by Repetition
    grouped = df.groupby("Repetition")
    all_feature_rows = []
    for rep_id, group_data in grouped:
        emg_array = group_data["EMG"].to_numpy()

        # Compute Wavelength-based features
        feats_dict = compute_emg_wavelength_features_for_repetition(
            emg_array, sampling_rate=sampling_rate, muscle_name=muscle_name
        )

        # Add identifying info
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id

        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)


def extract_all_wavelength_emg_features(
    base_directory="processed_data_35_i",
    muscle_folders=(
        "emg_deltoideus_anterior",
        "emg_deltoideus_posterior",
        "emg_infraspinatus",
        "emg_latissimus_dorsi",
        "emg_pectoralis_major",
        "emg_trapezius_ascendens",
    ),
    sampling_rate=1000,
    output_csv="Features/Extracted/wavelengthEMG_features_EMG.csv",
    metadata_file=None
):
    """
    Orchestrates the Wavelength (WV) feature extraction for all relevant muscles and subjects.
    Produces a single CSV with rows per [Subject, Repetition].

    Final CSV columns:
      [Subject, Repetition, <muscle_name>_WV_Total, <muscle_name>_WV_Per_Second, ...]

    Parameters:
      base_directory   : str  -> root folder with subfolders for each muscle
      muscle_folders  : list/tuple of str -> subfolder names for each muscle
      sampling_rate   : float -> EMG sampling rate
      output_csv      : str  -> path to save the final CSV
      metadata_file   : str or None -> optional path to subject-level metadata (merge on 'Subject')
    """
    import os

    all_dataframes = []

    # 1. Loop over muscle folders
    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Muscle folder not found: {muscle_path}. Skipping.")
            continue

        # 2. Locate CSV files
        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        # 3. Process each CSV
        for csv_path in csv_files:
            print(f"[INFO] Processing {csv_path}...")
            df_wv_feats = extract_emg_wavelength_features_from_file(
                csv_path, muscle_name=muscle_name, sampling_rate=sampling_rate
            )
            all_dataframes.append(df_wv_feats)

    if not all_dataframes:
        print("[INFO] No Wavelength EMG features extracted. Check directory structure.")
        return

    # 4. Concatenate all data
    combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # 5. Group by (Subject, Repetition) to unify multiple muscle columns
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # 6. Optionally merge with metadata (e.g., demographic info)
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # 7. Create directory if needed and save
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Wavelength-based EMG features saved to {output_csv}")
    print(f"[INFO] Final record count in {output_csv}: {len(wide_df)}")