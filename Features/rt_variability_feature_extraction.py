import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def compute_rt_variability_features_for_repetition(rep_df, sampling_rate, sensor, body_part):
    """
    Compute RT variability features for one repetition from the difference between consecutive samples.

    Parameters:
      rep_df : pd.DataFrame
          DataFrame with columns 'X', 'Y', 'Z' representing the raw signal in one repetition.
      sampling_rate : int or float
          Sensor sampling rate (Hz).
      sensor : str
          Sensor type (e.g., 'acc' or 'gyr').
      body_part : str
          Body part name (e.g., 'Shoulder') to embed in the feature names.

    Returns:
      feats : dict
          Dictionary with feature names containing sensor and body part, for example:
          "X_Shoulder_acc_rtVar_Mean", "Magnitude_Shoulder_acc_rtVar_Energy", etc.
    """
    feats = {}
    channels = ['X', 'Y', 'Z']

    # Process each channel for the rt variability features.
    for ch in channels:
        key_prefix = f"{ch}_{body_part}_{sensor}_rtVar"
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
            if len(data) < 2:
                # Not enough data to compute differences.
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
                diff_data = np.diff(data)
                feats[f"{key_prefix}_Mean"] = np.mean(diff_data)
                feats[f"{key_prefix}_Std"] = np.std(diff_data, ddof=1) if len(diff_data) > 1 else np.nan
                feats[f"{key_prefix}_Max"] = np.max(diff_data)
                feats[f"{key_prefix}_Min"] = np.min(diff_data)
                feats[f"{key_prefix}_Range"] = np.max(diff_data) - np.min(diff_data)
                feats[f"{key_prefix}_RMS"] = np.sqrt(np.mean(diff_data ** 2))
                feats[f"{key_prefix}_Energy"] = np.sum(diff_data ** 2)
                feats[f"{key_prefix}_IQR"] = np.percentile(diff_data, 75) - np.percentile(diff_data, 25)
                try:
                    feats[f"{key_prefix}_Skewness"] = skew(diff_data)
                    feats[f"{key_prefix}_Kurtosis"] = kurtosis(diff_data)
                except Exception:
                    feats[f"{key_prefix}_Skewness"] = np.nan
                    feats[f"{key_prefix}_Kurtosis"] = np.nan

    # Process combined magnitude for variability.
    mag_prefix = f"Magnitude_{body_part}_{sensor}_rtVar"
    if all(ch in rep_df.columns for ch in channels):
        x = rep_df["X"].to_numpy()
        y = rep_df["Y"].to_numpy()
        z = rep_df["Z"].to_numpy()
        magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if len(magnitude) < 2:
            feats[f"{mag_prefix}_Mean"] = np.nan
            feats[f"{mag_prefix}_Std"] = np.nan
            feats[f"{mag_prefix}_Max"] = np.nan
            feats[f"{mag_prefix}_Min"] = np.nan
            feats[f"{mag_prefix}_Range"] = np.nan
            feats[f"{mag_prefix}_RMS"] = np.nan
            feats[f"{mag_prefix}_Energy"] = np.nan
            feats[f"{mag_prefix}_IQR"] = np.nan
            feats[f"{mag_prefix}_Skewness"] = np.nan
            feats[f"{mag_prefix}_Kurtosis"] = np.nan
        else:
            diff_magnitude = np.diff(magnitude)
            feats[f"{mag_prefix}_Mean"] = np.mean(diff_magnitude)
            feats[f"{mag_prefix}_Std"] = np.std(diff_magnitude, ddof=1) if len(diff_magnitude) > 1 else np.nan
            feats[f"{mag_prefix}_Max"] = np.max(diff_magnitude)
            feats[f"{mag_prefix}_Min"] = np.min(diff_magnitude)
            feats[f"{mag_prefix}_Range"] = np.max(diff_magnitude) - np.min(diff_magnitude)
            feats[f"{mag_prefix}_RMS"] = np.sqrt(np.mean(diff_magnitude ** 2))
            feats[f"{mag_prefix}_Energy"] = np.sum(diff_magnitude ** 2)
            feats[f"{mag_prefix}_IQR"] = np.percentile(diff_magnitude, 75) - np.percentile(diff_magnitude, 25)
            try:
                feats[f"{mag_prefix}_Skewness"] = skew(diff_magnitude)
                feats[f"{mag_prefix}_Kurtosis"] = kurtosis(diff_magnitude)
            except Exception:
                feats[f"{mag_prefix}_Skewness"] = np.nan
                feats[f"{mag_prefix}_Kurtosis"] = np.nan
    else:
        for feat in ["Mean", "Std", "Max", "Min", "Range", "RMS", "Energy", "IQR", "Skewness", "Kurtosis"]:
            feats[f"{mag_prefix}_{feat}"] = np.nan

    # Optionally, add repetition duration as an auxiliary feature.
    feats[f"RepetitionDuration_{body_part}_{sensor}_rtVar"] = rep_df.shape[0] / sampling_rate

    return feats


def extract_rt_variability_features_from_file(csv_path, sensor, body_part, sampling_rate=100):
    """
    Reads a CSV file containing IMU data (expected columns: X, Y, Z, Repetition)
    and computes RT variability features for each repetition, integrating the sensor type
    and body part into the feature names.

    Parameters:
      csv_path : str
          Path to the CSV file.
      sensor : str
          Sensor type ('acc' or 'gyr').
      body_part : str
          Body part name (e.g., 'Shoulder').
      sampling_rate : int or float
          Sampling rate of the sensor (Hz).

    Returns:
      df_feats : pd.DataFrame
          A DataFrame with one row per repetition and features such as:
          X_Shoulder_acc_rtVar_Mean, Magnitude_Shoulder_acc_rtVar_Energy, etc.
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
        feats_dict = compute_rt_variability_features_for_repetition(rep_data, sampling_rate, sensor, body_part)
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id
        all_feature_rows.append(feats_dict)

    return pd.DataFrame(all_feature_rows)


def extract_all_rt_variability_features(
        base_directory="processed_data_35_i",
        body_parts=['Shoulder', 'Forearm', 'Pelvis', 'Upperarm', 'Torso', 'Palm'],
        sensors=['acc', 'gyr'],
        sampling_rate=100,
        output_csv="Features/Extracted/rt_variability_features_IMU.csv",
        metadata_file=None
):
    """
    Iterates over each body part and sensor type (acc and gyr), reads all CSV files
    (with columns X, Y, Z, Repetition) from the corresponding sensor folder,
    computes RT variability features (with feature names integrating sensor and body part)
    for each repetition, and consolidates the results into a single CSV.

    Parameters:
      base_directory : str
          Root directory containing subfolders for each body part.
      body_parts : list
          List of body parts (e.g., ['Shoulder', 'Forearm', ...]).
      sensors : list
          List of sensor types, typically ['acc', 'gyr'].
      sampling_rate : int or float
          Sensor sampling rate (Hz).
      output_csv : str
          Path to save the consolidated CSV.
      metadata_file : str or None
          Optional metadata CSV to merge based on "Subject".
    """
    all_dfs = []

    for bp in body_parts:
        for sensor in ["acc", "gyr"]:
            sensor_folder = os.path.join(base_directory, bp, sensor)
            if not os.path.isdir(sensor_folder):
                print(f"[WARN] Folder does not exist: {sensor_folder}. Skipping.")
                continue
            csv_files = glob.glob(os.path.join(sensor_folder, "*.csv"))
            if not csv_files:
                print(f"[WARN] No CSV files in {sensor_folder}. Skipping {bp} - {sensor}.")
                continue
            for csv_path in csv_files:
                print(f"[INFO] Processing {csv_path} ...")
                try:
                    df_feats = extract_rt_variability_features_from_file(csv_path, sensor, bp, sampling_rate)
                    all_dfs.append(df_feats)
                except Exception as e:
                    print(f"[ERROR] Processing {csv_path} failed: {e}")

    if not all_dfs:
        print("[INFO] No RT variability features extracted. Check your data paths.")
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
    print(f"[INFO] RT variability features saved to {output_csv}")

