import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def compute_gyr_features_for_repetition(rep_df, sampling_rate, body_part):
    """
    Compute 3-axis gyroscope (angular velocity) features for one repetition.
    The body part is integrated in the feature names.

    Parameters:
      rep_df : DataFrame
          DataFrame with columns 'X', 'Y', 'Z' for one repetition.
      sampling_rate : int or float
          Sampling rate of the sensor (Hz).
      body_part : str
          Name of the body part (e.g., 'Shoulder') to embed in feature names.

    Returns:
      feats : dict
          Dictionary with features named like:
            X_Shoulder_gyr_Mean, X_Shoulder_gyr_Std, ...,
            Magnitude_Shoulder_gyr_Mean, etc.
    """
    feats = {}
    channels = ['X', 'Y', 'Z']

    # Process each individual axis.
    for ch in channels:
        key_prefix = f"{ch}_{body_part}_gyr"
        if ch not in rep_df.columns:
            feats[f"{key_prefix}_Mean"] = np.nan
            feats[f"{key_prefix}_Std"] = np.nan
            feats[f"{key_prefix}_Max"] = np.nan
            feats[f"{key_prefix}_Min"] = np.nan
            feats[f"{key_prefix}_Range"] = np.nan
            feats[f"{key_prefix}_RMS"] = np.nan
            feats[f"{key_prefix}_Energy"] = np.nan
            feats[f"{key_prefix}_IQR"] = np.nan
            feats[f"{key_prefix}_Skewness"] = np.nan
            feats[f"{key_prefix}_Kurtosis"] = np.nan
        else:
            data = rep_df[ch].to_numpy()
            feats[f"{key_prefix}_Mean"] = np.mean(data)
            feats[f"{key_prefix}_Std"] = np.std(data, ddof=1) if data.size > 1 else np.nan
            feats[f"{key_prefix}_Max"] = np.max(data)
            feats[f"{key_prefix}_Min"] = np.min(data)
            feats[f"{key_prefix}_Range"] = np.max(data) - np.min(data)
            feats[f"{key_prefix}_RMS"] = np.sqrt(np.mean(data**2))
            feats[f"{key_prefix}_Energy"] = np.sum(data**2)
            feats[f"{key_prefix}_IQR"] = np.percentile(data, 75) - np.percentile(data, 25)
            feats[f"{key_prefix}_Skewness"] = skew(data)
            feats[f"{key_prefix}_Kurtosis"] = kurtosis(data)

    # Process combined magnitude from X, Y, Z:
    mag_prefix = f"Magnitude_{body_part}_gyr"
    if all(ch in rep_df.columns for ch in channels):
        x = rep_df["X"].to_numpy()
        y = rep_df["Y"].to_numpy()
        z = rep_df["Z"].to_numpy()
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        feats[f"{mag_prefix}_Mean"] = np.mean(magnitude)
        feats[f"{mag_prefix}_Std"] = np.std(magnitude, ddof=1) if magnitude.size > 1 else np.nan
        feats[f"{mag_prefix}_Max"] = np.max(magnitude)
        feats[f"{mag_prefix}_Min"] = np.min(magnitude)
        feats[f"{mag_prefix}_Range"] = np.max(magnitude) - np.min(magnitude)
        feats[f"{mag_prefix}_RMS"] = np.sqrt(np.mean(magnitude**2))
        feats[f"{mag_prefix}_Energy"] = np.sum(magnitude**2)
        feats[f"{mag_prefix}_IQR"] = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
        feats[f"{mag_prefix}_Skewness"] = skew(magnitude)
        feats[f"{mag_prefix}_Kurtosis"] = kurtosis(magnitude)
    else:
        for feat in ["Mean", "Std", "Max", "Min", "Range", "RMS", "Energy", "IQR", "Skewness", "Kurtosis"]:
            feats[f"{mag_prefix}_{feat}"] = np.nan

    # Add repetition duration (in seconds) as a feature.
    feats[f"RepetitionDuration_{body_part}_gyr"] = rep_df.shape[0] / sampling_rate

    return feats

def extract_gyr_3axis_features_from_file(csv_path, body_part, sampling_rate=100):
    """
    Reads a CSV file containing IMU gyroscope data (expected columns: X, Y, Z, Repetition)
    and computes 3-Axis Angular Velocity features for each repetition.
    Integrated feature names include the body part.

    Parameters:
      csv_path : str
          Path to the CSV file.
      body_part : str
          The body part (e.g., 'Shoulder') to embed in the feature names.
      sampling_rate : int or float
          The sensor sampling rate (Hz).

    Returns:
      df_feats : DataFrame
          A DataFrame with one row per repetition and feature names like:
          X_Shoulder_gyr_Mean, Magnitude_Shoulder_gyr_Energy, etc.
    """
    filename = os.path.basename(csv_path)
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"X", "Y", "Z", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} is missing required columns: {required_cols - set(df.columns)}")

    all_feature_rows = []
    for rep_id, rep_data in df.groupby("Repetition"):
        feats_dict = compute_gyr_features_for_repetition(rep_data, sampling_rate, body_part)
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id
        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)

def extract_all_gyr_3axis_features(
        base_directory="processed_data_35_i",
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sampling_rate=100,
        output_csv="Features/Extracted/gyr_3axis_features_IMU.csv",
        metadata_file=None
):
    """
    Iterates over each body part (processing only the 'gyr' folder), reads
    all gyroscope CSV files (with columns X, Y, Z, Repetition),
    computes 3-Axis Angular Velocity features (with integrated body part in the feature names)
    for each repetition, and consolidates them into a single CSV.

    Parameters:
      base_directory : str
          Root directory containing subfolders for each body part.
      body_parts : list
          List of body parts (e.g., ['Shoulder', 'Forearm', ...]).
      sampling_rate : int or float
          Gyroscope sampling rate in Hz.
      output_csv : str
          File path for the consolidated CSV.
      metadata_file : str or None
          Optional CSV file with subject metadata merged based on "Subject".
    """
    all_dfs = []

    for bp in body_parts:
        sensor_folder = os.path.join(base_directory, bp, "gyr")
        if not os.path.isdir(sensor_folder):
            print(f"[WARN] Folder does not exist: {sensor_folder}. Skipping.")
            continue

        csv_files = glob.glob(os.path.join(sensor_folder, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {sensor_folder}. Skipping {bp}.")
            continue

        for csv_path in csv_files:
            print(f"[INFO] Processing {csv_path} ...")
            try:
                df_feats = extract_gyr_3axis_features_from_file(csv_path, bp, sampling_rate)
                all_dfs.append(df_feats)
            except Exception as e:
                print(f"[ERROR] Processing {csv_path} failed: {e}")

    if not all_dfs:
        print("[INFO] No 3-Axis Angular Velocity features extracted. Check your data paths.")
        return

    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] 3-Axis Angular Velocity features saved to {output_csv}")

