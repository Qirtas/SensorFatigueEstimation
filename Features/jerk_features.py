import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter
from scipy.stats import skew, kurtosis


def compute_jerk_features_for_repetition(acc_data, dt, body_part):
    """
    Compute jerk features (X, Y, Z, magnitude) for a single repetition
    and embed the body_part in each feature name.

    Parameters
    ----------
    acc_data : pandas.DataFrame
        A DataFrame containing columns [X, Y, Z] for the repetition.
    dt : float
        Sampling interval (1 / sampling_rate).
    body_part : str
        Name of the body part (e.g., 'Shoulder') to embed in feature columns.

    Returns
    -------
    feature_dict : dict
        Dictionary of computed features (mean, median, max, min, skew, kurtosis, etc.)
        for X, Y, Z, and magnitude of jerk, with body_part appended to column names.
        Example keys: "jerkX_mean_Shoulder", "jerkMag_kurtosis_Shoulder", etc.
    """
    # Convert to numpy arrays
    ax = acc_data['X'].to_numpy()
    ay = acc_data['Y'].to_numpy()
    az = acc_data['Z'].to_numpy()

    # --- Compute jerk for each axis (finite difference) ---
    # jerk_x[t] = (ax[t+1] - ax[t]) / dt
    jerk_x = np.diff(ax) / dt
    jerk_y = np.diff(ay) / dt
    jerk_z = np.diff(az) / dt

    # --- Jerk magnitude ---
    jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)

    # --- Descriptive statistics function ---
    def describe(signal, prefix):
        """
        prefix examples: 'jerkX', 'jerkY', 'jerkZ', 'jerkMag'
        """
        stats_dict = {}
        if len(signal) == 0:
            # Handle edge case if repetition has <2 samples
            stats_dict[f"{prefix}_mean_{body_part}"] = np.nan
            stats_dict[f"{prefix}_median_{body_part}"] = np.nan
            stats_dict[f"{prefix}_min_{body_part}"] = np.nan
            stats_dict[f"{prefix}_max_{body_part}"] = np.nan
            stats_dict[f"{prefix}_skew_{body_part}"] = np.nan
            stats_dict[f"{prefix}_kurtosis_{body_part}"] = np.nan
            stats_dict[f"{prefix}_std_{body_part}"] = np.nan
            stats_dict[f"{prefix}_rms_{body_part}"] = np.nan
            stats_dict[f"{prefix}_range_{body_part}"] = np.nan
            return stats_dict

        # Basic stats
        stats_dict[f"{prefix}_mean_{body_part}"] = np.mean(signal)
        stats_dict[f"{prefix}_median_{body_part}"] = np.median(signal)
        stats_dict[f"{prefix}_min_{body_part}"] = np.min(signal)
        stats_dict[f"{prefix}_max_{body_part}"] = np.max(signal)
        stats_dict[f"{prefix}_skew_{body_part}"] = skew(signal) if len(signal) > 1 else 0.0
        stats_dict[f"{prefix}_kurtosis_{body_part}"] = kurtosis(signal) if len(signal) > 1 else 0.0

        # Additional stats often used in motion analysis
        stats_dict[f"{prefix}_std_{body_part}"] = np.std(signal, ddof=1)  # sample std
        stats_dict[f"{prefix}_rms_{body_part}"] = np.sqrt(np.mean(signal ** 2))
        stats_dict[f"{prefix}_range_{body_part}"] = np.max(signal) - np.min(signal)

        return stats_dict

    feature_dict = {}

    # Axis-specific jerk features
    feature_dict.update(describe(jerk_x, "jerkX"))
    feature_dict.update(describe(jerk_y, "jerkY"))
    feature_dict.update(describe(jerk_z, "jerkZ"))

    # Magnitude-based jerk features
    feature_dict.update(describe(jerk_magnitude, "jerkMag"))

    return feature_dict


def extract_jerk_features_from_file(csv_path, sampling_rate, body_part):
    """
    Reads one accelerometer CSV for a subject & body part,
    computes jerk features per repetition, and returns a DataFrame with:
    Subject, Repetition, (all jerk features w/ body_part embedded in name)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns [X, Y, Z, Repetition].
    sampling_rate : float
        IMU sampling frequency (e.g., 100 Hz).
    body_part : str
        Name of the body part (e.g., 'Shoulder').

    Returns
    -------
    features_df : pandas.DataFrame
        DataFrame with columns: [Subject, Repetition, jerkX_mean_<body_part>, jerkX_std_<body_part>, ...].
        One row per (Subject, Repetition).
    """
    # Extract subject ID from filename
    filename = os.path.basename(csv_path)
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = {'X', 'Y', 'Z', 'Repetition'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing required columns: {required_cols - set(df.columns)}.")

    dt = 1.0 / sampling_rate

    # Group by Repetition
    grouped = df.groupby('Repetition')

    all_features = []
    for rep_id, group_data in grouped:
        # Compute jerk features
        jerk_stats = compute_jerk_features_for_repetition(group_data, dt, body_part)

        # Include identifier columns
        jerk_stats['Subject'] = subject_id
        jerk_stats['Repetition'] = rep_id
        all_features.append(jerk_stats)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    return features_df


def extract_all_jerk_features(
        base_directory="processed_data_35_i",
        body_parts=('Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm'),
        sampling_rate=100,
        output_csv="OutputCSVFiles/consolidated_jerk_features.csv",
        metadata_file=None
):
    """
    Iterates over body parts, reads each CSV in 'acc' folders,
    computes jerk features with body_part embedded in column names,
    and pivots data so each row = one repetition (per subject),
    containing columns for all body parts.

    Parameters
    ----------
    base_directory : str
        Directory containing body-part subfolders (e.g., 'processed_data_35_i').
    body_parts : tuple or list of str
        Body parts to process, each with an 'acc' subfolder.
    sampling_rate : float
        IMU sampling frequency in Hz.
    output_csv : str
        Path to save the consolidated wide-format CSV.
    metadata_file : str or None
        Optional path to a CSV file with additional subject-level info
        (merged on 'Subject'). Set to None if not using.
    """
    # 1. Collect features from all subjects & body parts in a "long" DataFrame
    #    i.e., columns: [Subject, Repetition, <jerk features>, BodyPart?].
    #    Actually, we are embedding the body part in feature names already,
    #    so we just need [Subject, Repetition, <all features for that body part>].
    all_dfs = []

    for bp in body_parts:
        acc_folder = os.path.join(base_directory, bp, "acc")
        if not os.path.isdir(acc_folder):
            print(f"Warning: {acc_folder} not found; skipping body part '{bp}'.")
            continue

        csv_files = sorted(glob.glob(os.path.join(acc_folder, "*.csv")))
        if not csv_files:
            print(f"No CSV files in {acc_folder}; skipping.")
            continue

        for csv_path in csv_files:
            features_df = extract_jerk_features_from_file(csv_path, sampling_rate, bp)
            all_dfs.append(features_df)

    if not all_dfs:
        print("No jerk features extracted. Please check your data paths.")
        return

    # 2. Concatenate all into one DataFrame
    #    Currently, each DF has columns: [Subject, Repetition, jerkX_mean_<bp>, jerkX_median_<bp>, ...]
    #    Possibly multiple rows for the same Subject+Repetition but *different* body parts
    #    are each in separate DataFrames. We want to combine them "wide" so each row has
    #    features for all body parts for that Subject+Repetition.
    combined_long_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # Because we embedded the body part in each feature name, we don't actually
    # need to pivot on "BodyPart" anymore. All columns are already unique
    # (like jerkX_mean_Shoulder, jerkX_mean_Forearm, etc.).
    # We just need to "merge" by (Subject, Repetition).
    # The easiest way is group by Subject+Repetition and combine columns.
    # However, as we did that *within* each sub-DataFrame, there's exactly 1 row
    # for each (Subject, Repetition, body_part). So let's do a "groupby" and join them.
    # But an easier method is to do:
    #    combined_long_df = combined_long_df.groupby(['Subject','Repetition'], as_index=False).first()
    # Actually, that won't suffice because we have multiple sets of columns
    # from different body parts. We'll need a proper merge or pivot.

    # Approach: We'll pivot so that Subject+Repetition is the index, and
    # we combine the columns for each row. But because the columns are already
    # unique for each body part, we just need one row per (Subject, Repetition).
    # So let's do a groupby-aggregation that picks the first non-null from each column.

    # Step 1: set index
    combined_long_df.set_index(["Subject", "Repetition"], inplace=True)
    # Step 2: groupby index level. For each group, pick the first non-null values
    # across columns. This effectively merges multiple body-part rows into one wide row.
    wide_df = combined_long_df.groupby(level=["Subject", "Repetition"]).first()

    # Now we have one row per (Subject, Repetition), with columns for all features
    # from all body parts (since they're uniquely named).
    wide_df.reset_index(inplace=True)

    # 3. Optionally merge with metadata (on "Subject") if provided
    if metadata_file is not None and os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, metadata_df, on="Subject", how="left")

    # 4. Save final CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] Jerk features saved to {output_csv}")


def interpolate_missing_shoulder_features(file_path, output_path):
    """
    Interpolate missing Shoulder-related features for Subject 3 in the given dataset.

    Parameters:
    ----------
    file_path : str
        Path to the input CSV file containing jerk features.
    output_path : str
        Path to save the updated CSV file with interpolated features.

    Returns:
    -------
    None
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Identify Shoulder-related features
    shoulder_features = [col for col in data.columns if "Shoulder" in col]

    # Filter data for Subject 3
    subject_3_data = data[data["Subject"] == 3]

    # Interpolate missing Shoulder features for Subject 3
    interpolated_features = subject_3_data[shoulder_features].interpolate(
        method="linear", limit_direction="forward", axis=0
    )

    # Replace the missing Shoulder features with interpolated values
    data.loc[data["Subject"] == 3, shoulder_features] = interpolated_features

    # Save the updated dataset
    data.to_csv(output_path, index=False)

    print(f"Interpolation completed. Updated file saved to: {output_path}")

def print_missing_values(file_path):
    data = pd.read_csv(file_path)
    rows, columns = data.shape
    print(f"Total Rows: {rows}")
    print(f"Total Columns: {columns}")

    # Check for missing or null values
    missing_values = data.isnull().sum()

    # Print columns with missing values
    print("Columns with missing values:")
    print(missing_values[missing_values > 0])

    # Total missing values
    total_missing = missing_values.sum()
    print(f"Total Missing Values: {total_missing}")


import pandas as pd

# def fill_missing_values(csv_path, output_path=None, fill_method='both'):
#     """
#     Fill missing values in a CSV file using forward or backward fill.
#
#     Parameters:
#     - csv_path (str): Path to the input CSV file.
#     - output_path (str): Path to save the updated CSV file (optional).
#     - fill_method (str): Method to fill missing values. Options:
#         'forward' - Forward fill (propagate last valid value forward).
#         'backward' - Backward fill (propagate next valid value backward).
#         'both' - Apply forward fill first, then backward fill for remaining missing values.
#
#     Returns:
#     - pd.DataFrame: Updated DataFrame with missing values filled.
#     """
#     # Load the CSV file into a DataFrame
#     df = pd.read_csv(csv_path)
#
#     # Count missing values before filling
#     missing_before = df.isnull().sum().sum()
#     print(f"Total missing values before filling: {missing_before}")
#
#     # Fill missing values based on the specified method
#     if fill_method == 'forward':
#         df.fillna(method='ffill', inplace=True)
#     elif fill_method == 'backward':
#         df.fillna(method='bfill', inplace=True)
#     elif fill_method == 'both':
#         df.fillna(method='ffill', inplace=True)
#         df.fillna(method='bfill', inplace=True)
#     else:
#         raise ValueError("Invalid fill_method. Use 'forward', 'backward', or 'both'.")
#
#     # Count missing values after filling
#     missing_after = df.isnull().sum().sum()
#     print(f"Total missing values after filling: {missing_after}")
#
#     # Save the updated DataFrame to a new CSV file if output_path is specified
#     if output_path:
#         df.to_csv(output_path, index=False)
#         print(f"Updated CSV file saved to: {output_path}")
#
#     return df


def fill_missing_values(csv_path, output_path=None, fill_method='both'):
    """
    Fill missing values in CSV files using forward or backward fill.

    Parameters:
    - csv_path (str or list): Path to single CSV file or list of CSV files
    - output_path (str): Path to save the updated CSV files (optional)
    - fill_method (str): Method to fill missing values ('forward', 'backward', or 'both')

    Returns:
    - pd.DataFrame or list: Updated DataFrame(s) with missing values filled
    """
    # Handle single string path or list of paths
    if isinstance(csv_path, str):
        paths = [csv_path]
    else:
        paths = csv_path

    processed_dfs = []

    for path in paths:
        try:
            # Load the CSV file
            df = pd.read_csv(path)
            filename = os.path.basename(path)
            print(f"\nProcessing file: {filename}")

            # Count missing values before filling
            missing_before = df.isnull().sum().sum()
            print(f"Total missing values before filling: {missing_before}")

            # Show columns with missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            # if missing_cols:
            #     print("Columns with missing values:")
            #     for col in missing_cols:
            #         print(f"- {col}: {df[col].isnull().sum()} missing values")

            # Fill missing values
            if fill_method == 'forward':
                df.fillna(method='ffill', inplace=True)
            elif fill_method == 'backward':
                df.fillna(method='bfill', inplace=True)
            elif fill_method == 'both':
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
            else:
                raise ValueError("Invalid fill_method. Use 'forward', 'backward', or 'both'.")

            # Count missing values after filling
            missing_after = df.isnull().sum().sum()
            print(f"Total missing values after filling: {missing_after}")

            # Save the updated DataFrame if output_path is specified
            if output_path:
                # Create output filename with '_filled' suffix
                base_name = os.path.splitext(filename)[0]
                new_filename = f"{base_name}_filled.csv"
                save_path = os.path.join(output_path, new_filename)

                # Create output directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)

                df.to_csv(save_path, index=False)
                print(f"Saved to: {save_path}")

            processed_dfs.append(df)

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue

    # Return single DataFrame if input was single path, otherwise return list
    return processed_dfs[0] if len(processed_dfs) == 1 else processed_dfs


# Example usage:
# updated_df = fill_missing_values(
#     csv_path="final_features.csv",
#     output_path="final_features_filled.csv",
#     fill_method='both'
# )
