import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import skew, kurtosis, linregress
from scipy.integrate import cumulative_trapezoid

# Configuration Parameters

# Sampling rate in Hz (adjust based on your dataset)
SAMPLING_RATE = 100  # Example: 100 Hz
DELTA_T = 1 / SAMPLING_RATE

# Base directory containing the processed data
BASE_DIRECTORY = "processed_data_35_i"

# List of body parts to process (extend as needed)
BODY_PARTS = ['Forearm', 'Pelvis', 'Palm', 'Shoulder', 'Torso', 'Upperarm']

# Output CSV file name
OUTPUT_JERK_CSV = "OutputCSVFiles/consolidated_jerk_features.csv"
OUTPUT_FREQ_CSV = "OutputCSVFiles/consolidated_movFreq_features.csv"
OUTPUT_RTV_CSV = "OutputCSVFiles/consolidated_RTV_features.csv"
OUTPUT_MPSD_CSV = "OutputCSVFiles/consolidated_mpsd_features.csv"
OUTPUT_ROM_CSV = "OutputCSVFiles/consolidated_ROM_features.csv"
MERGED_CSV = "OutputCSVFiles/Merged.csv"


def compute_jerk(acc_x, acc_y, acc_z, delta_t):
    """
    Compute the jerk magnitude given acceleration data along X, Y, Z axes.

    Parameters:
    - acc_x (np.ndarray): Acceleration data along X-axis.
    - acc_y (np.ndarray): Acceleration data along Y-axis.
    - acc_z (np.ndarray): Acceleration data along Z-axis.
    - delta_t (float): Time interval between consecutive samples.

    Returns:
    - jerk_magnitude (np.ndarray): Computed jerk magnitudes.
    """
    # Compute the time derivative (jerk) for each axis
    jerk_x = np.diff(acc_x) / delta_t
    jerk_y = np.diff(acc_y) / delta_t
    jerk_z = np.diff(acc_z) / delta_t

    # Compute the jerk magnitude
    jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

    return jerk_magnitude


def extract_jerk_features(jerk_magnitude):
    """
    Extract statistical features from jerk magnitude.

    Parameters:
    - jerk_magnitude (np.ndarray): Computed jerk magnitudes.

    Returns:
    - features (dict): Dictionary containing mean and standard deviation of jerk.
    """
    jerk_mean = np.mean(jerk_magnitude)
    jerk_std = np.std(jerk_magnitude)
    jerk_median = np.median(jerk_magnitude)
    jerk_skew = pd.Series(jerk_magnitude).skew()
    jerk_kurtosis = pd.Series(jerk_magnitude).kurtosis()
    jerk_min = np.min(jerk_magnitude)
    jerk_max = np.max(jerk_magnitude)
    jerk_range = jerk_max - jerk_min
    jerk_iqr = np.percentile(jerk_magnitude, 75) - np.percentile(jerk_magnitude, 25)

    return {'jerk_mean': jerk_mean, 'jerk_std': jerk_std,
        'jerk_median': jerk_median,
        'jerk_skew': jerk_skew,
        'jerk_kurtosis': jerk_kurtosis,
        'jerk_min': jerk_min,
        'jerk_max': jerk_max,
        'jerk_range': jerk_range,
        'jerk_iqr': jerk_iqr
    }


def process_acc_file(acc_file, body_part, delta_t, feature_dict):
    """
    Process a single acceleration CSV file to extract jerk features.

    Parameters:
    - acc_file (str): Path to the acceleration CSV file.
    - body_part (str): Name of the body part (e.g., 'Forearm').
    - delta_t (float): Time interval between consecutive samples.
    - feature_dict (dict): Dictionary to store extracted features.

    Returns:
    - None (updates feature_dict in place)
    """
    # Extract Subject ID from the filename
    filename = os.path.basename(acc_file)
    parts = filename.split('_')
    try:
        subject_id = int(parts[1])
    except (IndexError, ValueError):
        print(f"Filename {filename} does not match the expected format. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(acc_file)
    except Exception as e:
        print(f"Error reading {acc_file}: {e}")
        return

    # Verify required columns exist
    required_columns = {'X', 'Y', 'Z', 'Repetition', 'borg'}
    if not required_columns.issubset(df.columns):
        print(f"File {acc_file} is missing required columns. Skipping.")
        return

    # Group data by Repetition
    repetitions = df['Repetition'].unique()

    for rep in repetitions:
        rep_df = df[df['Repetition'] == rep].reset_index(drop=True)

        # Extract X, Y, Z accelerations
        acc_x = rep_df['X'].values
        acc_y = rep_df['Y'].values
        acc_z = rep_df['Z'].values

        # Check if there's enough data to compute jerk
        if len(acc_x) < 2:
            print(f"Not enough data points in Subject {subject_id}, Repetition {rep} for body part {body_part}. Skipping.")
            continue

        # Compute jerk magnitude
        jerk_magnitude = compute_jerk(acc_x, acc_y, acc_z, delta_t)

        # Extract features
        jerk_features = extract_jerk_features(jerk_magnitude)

        # Retrieve the Borg scale value (assuming it's constant per repetition)
        borg_value = rep_df['borg'].iloc[0]

        # Initialize a unique key for each Subject and Repetition
        key = (subject_id, rep)

        if key not in feature_dict:
            feature_dict[key] = {'Subject_ID': subject_id, 'Repetition': rep, 'borg': borg_value}

        # Define feature names
        feature_mean_name = f"jerk_{body_part.lower()}_mean"
        feature_std_name = f"jerk_{body_part.lower()}_std"
        feature_median_name = f"jerk_{body_part.lower()}_median"
        feature_skew_name = f"jerk_{body_part.lower()}_skew"
        feature_kurtosis_name = f"jerk_{body_part.lower()}_kurtosis"
        feature_min_name = f"jerk_{body_part.lower()}_min"
        feature_max_name = f"jerk_{body_part.lower()}_max"
        feature_range_name = f"jerk_{body_part.lower()}_range"
        feature_iqr_name = f"jerk_{body_part.lower()}_iqr"

        # Assign the computed features
        feature_dict[key][feature_mean_name] = jerk_features['jerk_mean']
        feature_dict[key][feature_std_name] = jerk_features['jerk_std']
        feature_dict[key][feature_median_name] = jerk_features['jerk_median']
        feature_dict[key][feature_skew_name] = jerk_features['jerk_skew']
        feature_dict[key][feature_kurtosis_name] = jerk_features['jerk_kurtosis']
        feature_dict[key][feature_min_name] = jerk_features['jerk_min']
        feature_dict[key][feature_max_name] = jerk_features['jerk_max']
        feature_dict[key][feature_range_name] = jerk_features['jerk_range']
        feature_dict[key][feature_iqr_name] = jerk_features['jerk_iqr']


def extract_all_jerk_features(base_directory=BASE_DIRECTORY,
                               body_parts=BODY_PARTS,
                               sampling_rate=SAMPLING_RATE,
                               output_csv=OUTPUT_JERK_CSV,
                               metadata_file=None):
    """
    Extract jerk features from all acceleration CSV files and consolidate them into a single CSV.

    Parameters:
    - base_directory (str): Base directory containing the processed data.
    - body_parts (list): List of body parts to process.
    - sampling_rate (int): Sampling rate in Hz.
    - output_csv (str): Path to save the consolidated CSV.
    - metadata_file (str or None): Path to metadata CSV file. Set to None if not using.

    Returns:
    - None (saves the consolidated CSV to output_csv)
    """
    delta_t = 1 / sampling_rate

    # Initialize feature dictionary
    feature_dict = {}

    # Iterate through each body part
    for body_part in tqdm(body_parts, desc="Processing Body Parts"):
        acc_dir = os.path.join(base_directory, body_part, 'acc')

        if not os.path.isdir(acc_dir):
            print(f"Acceleration directory {acc_dir} does not exist. Skipping body part {body_part}.")
            continue

        # Pattern to match acceleration CSV files
        file_pattern = os.path.join(acc_dir, f"Subject_*_{body_part}_acc_with_borg_acc.csv")
        acc_files = glob.glob(file_pattern)

        if not acc_files:
            print(f"No acceleration files found in {acc_dir} for body part {body_part}.")
            continue

        # Process each acceleration file
        for acc_file in tqdm(acc_files, desc=f"Processing {body_part} Files", leave=False):
            process_acc_file(acc_file, body_part, delta_t, feature_dict)

    # Convert feature dictionary to DataFrame
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # Reset index to have Subject_ID and Repetition as columns
    feature_df.reset_index(drop=True, inplace=True)

    # Optional: Merge with metadata if available
    if metadata_file and os.path.exists(metadata_file):
        try:
            metadata_df = pd.read_csv(metadata_file)
            feature_df = feature_df.merge(metadata_df, on='Subject_ID', how='left')
            print(f"Merged metadata from {metadata_file}.")
        except Exception as e:
            print(f"Error merging metadata: {e}")

    # Save the consolidated DataFrame to CSV
    try:
        feature_df = feature_df.sort_values(by=['Subject_ID', 'Repetition']).reset_index(drop=True)
        feature_df.to_csv(output_csv, index=False)
        print(f"Consolidated jerk features saved to {output_csv}")
    except Exception as e:
        print(f"Error saving consolidated CSV: {e}")


def process_gyro_file(gyro_file, body_part, delta_t, feature_dict):
    """
    Process a single gyroscope file, extract movement frequency features, and handle missing Borg scale values.

    Parameters:
    - gyro_file (str): Path to the gyroscope CSV file.
    - body_part (str): Name of the body part.
    - delta_t (float): Time interval between consecutive samples.
    - feature_dict (dict): Dictionary to store movement frequency features.

    Returns:
    - None (updates feature_dict in place)
    """
    # Extract Subject ID from the filename
    filename = os.path.basename(gyro_file)
    parts = filename.split('_')
    try:
        subject_id = int(parts[1])
    except (IndexError, ValueError):
        print(f"Filename {filename} does not match the expected format. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(gyro_file)
    except Exception as e:
        print(f"Error reading {gyro_file}: {e}")
        return

    # Verify required columns exist
    required_columns = {'X', 'Y', 'Z', 'Repetition', 'borg'}
    if not required_columns.issubset(df.columns):
        print(f"File {gyro_file} is missing required columns. Skipping.")
        return

    # Fill missing Borg values for final repetitions
    if df['borg'].isnull().any():
        max_borg = df['borg'].max()
        if not np.isnan(max_borg):
            # Fill missing Borg values with 19 (max Borg scale)
            df['borg'].fillna(19, inplace=True)
            print(f"Missing Borg values in {gyro_file} filled with 19.")
        else:
            print(f"Missing Borg values in {gyro_file} and no valid max value found. Assigning NaN.")
            df['borg'].fillna(np.nan, inplace=True)

    # Group data by Repetition
    repetitions = df['Repetition'].unique()

    for rep in repetitions:
        rep_df = df[df['Repetition'] == rep].reset_index(drop=True)

        # Extract the Borg scale value
        if rep_df.empty or 'borg' not in rep_df.columns:
            borg_value = np.nan
            print(f"Missing 'borg' value for Subject {subject_id}, Repetition {rep}. Assigning NaN.")
        else:
            borg_value = rep_df['borg'].iloc[0]

        # Initialize a unique key for each Subject and Repetition
        key = (subject_id, rep, borg_value)

        # Always add the repetition to the feature dictionary
        if key not in feature_dict:
            feature_dict[key] = {'Subject_ID': subject_id, 'Repetition': rep, 'borg': borg_value}

        # Extract axis data
        axis_x = rep_df['X'].values if 'X' in rep_df else np.array([])
        axis_y = rep_df['Y'].values if 'Y' in rep_df else np.array([])
        axis_z = rep_df['Z'].values if 'Z' in rep_df else np.array([])

        # Initialize movement frequency values as NaN
        mov_freq_x, mov_freq_y, mov_freq_z, mov_freq_combined = np.nan, np.nan, np.nan, np.nan

        # Compute movement frequency if there are enough data points
        if len(axis_x) > 1:
            mov_freq_x = compute_movement_frequency(axis_x, delta_t)
            mov_freq_y = compute_movement_frequency(axis_y, delta_t)
            mov_freq_z = compute_movement_frequency(axis_z, delta_t)
            mov_freq_combined = max(mov_freq_x, mov_freq_y, mov_freq_z)

        # Assign movement frequency values (or NaN) to the feature dictionary
        feature_dict[key][f"movfreq_{body_part.lower()}_x"] = mov_freq_x
        feature_dict[key][f"movfreq_{body_part.lower()}_y"] = mov_freq_y
        feature_dict[key][f"movfreq_{body_part.lower()}_z"] = mov_freq_z
        feature_dict[key][f"movfreq_{body_part.lower()}_combined"] = mov_freq_combined



def compute_movement_frequency(signal, delta_t):
    """
    Compute the movement frequency for a given signal.

    Parameters:
    - signal (np.ndarray): 1D array of the signal.
    - delta_t (float): Time interval between consecutive samples.

    Returns:
    - float: Movement frequency value.
    """
    if len(signal) <= 1:
        return np.nan

    # Compute the FFT of the signal
    freqs = np.fft.rfftfreq(len(signal), delta_t)
    power_spectrum = np.abs(np.fft.rfft(signal)) ** 2

    # Find the frequency with the maximum power
    max_freq = freqs[np.argmax(power_spectrum)]
    return max_freq


def extract_movement_frequency_features(frequency):
    """
    Extract statistical features from movement frequency.

    Parameters:
    - frequency (float): Computed movement frequency in Hz.

    Returns:
    - features (dict): Dictionary containing frequency features.
    """
    # Since frequency is a single value per repetition, statistical features aren't directly applicable.
    # However, if you have multiple frequency measurements per repetition, you can compute statistics.
    # For this example, we'll assume a single frequency value per repetition.

    # To align with the jerk feature extraction, we'll compute derived features if possible.
    # Here, we'll just return the frequency as a feature.

    return {'movement_frequency': frequency}


def extract_all_movement_frequency_features(base_directory=BASE_DIRECTORY,
                                            body_parts=BODY_PARTS,
                                            sampling_rate=SAMPLING_RATE,
                                            output_csv=OUTPUT_FREQ_CSV):
    """
    Extract movement frequency features from all gyroscope CSV files and consolidate them into a single CSV.

    Parameters:
    - base_directory (str): Base directory containing the processed data.
    - body_parts (list): List of body parts to process.
    - sampling_rate (int): Sampling rate in Hz.
    - output_csv (str): Path to save the consolidated CSV.

    Returns:
    - None (saves the consolidated CSV to output_csv)
    """
    delta_t = 1 / sampling_rate

    # Initialize feature dictionary
    feature_dict = {}

    # Iterate through each body part
    for body_part in tqdm(body_parts, desc="Processing Body Parts"):
        gyro_dir = os.path.join(base_directory, body_part, 'gyr')

        if not os.path.isdir(gyro_dir):
            print(f"Gyroscope directory {gyro_dir} does not exist. Skipping body part {body_part}.")
            continue

        # Pattern to match gyroscope CSV files
        file_pattern = os.path.join(gyro_dir, f"Subject_*_{body_part}_gyr_with_borg_gyr.csv")
        gyro_files = glob.glob(file_pattern)

        if not gyro_files:
            print(f"No gyroscope files found in {gyro_dir} for body part {body_part}.")
            continue

        # Process each gyroscope file
        for gyro_file in tqdm(gyro_files, desc=f"Processing {body_part} Gyroscope Files", leave=False):
            process_gyro_file(gyro_file, body_part, delta_t, feature_dict)

    # Convert feature dictionary to DataFrame
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # Reset index to have Subject_ID and Repetition as columns
    feature_df.reset_index(drop=True, inplace=True)

    # Save the consolidated DataFrame to CSV
    try:
        feature_df = feature_df.sort_values(by=['Subject_ID', 'Repetition']).reset_index(drop=True)
        feature_df.to_csv(output_csv, index=False)
        print(f"\nConsolidated movement frequency features saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving consolidated CSV: {e}")


def compute_repetition_duration(rep_length, delta_t):
    """
    Compute the duration of a repetition.

    Parameters:
    - rep_length (int): Number of samples in the repetition.
    - delta_t (float): Time interval between consecutive samples.

    Returns:
    - duration (float): Duration of the repetition in seconds.
    """
    return rep_length * delta_t


def extract_repetition_time_variability_features(df, window_size=3):
    """
    Extract RTV statistical features using a moving window approach.

    Parameters:
    - df (pd.DataFrame): DataFrame containing repetition durations sorted by Subject_ID and Repetition.
    - window_size (int): Number of repetitions to include in the moving window.

    Returns:
    - df (pd.DataFrame): DataFrame with RTV features added.
    """
    df = df.sort_values(['Subject_ID', 'Repetition']).reset_index(drop=True)

    # Initialize RTV feature columns
    df['rtv_mean'] = np.nan
    df['rtv_std'] = np.nan
    df['rtv_median'] = np.nan
    df['rtv_skew'] = np.nan
    df['rtv_kurtosis'] = np.nan
    df['rtv_min'] = np.nan
    df['rtv_max'] = np.nan
    df['rtv_range'] = np.nan
    df['rtv_iqr'] = np.nan

    # Group by Subject_ID
    grouped = df.groupby('Subject_ID')

    for subject_id, group in grouped:
        group = group.sort_values('Repetition').reset_index(drop=True)
        for idx in group.index:
            # Define the window for RTV computation
            start_idx = max(idx - window_size + 1, 0)
            window = group.loc[start_idx:idx, 'repetition_duration']

            # Compute RTV features
            if len(window) >= 2:
                rtv_mean = window.mean()
                rtv_std = window.std()
                rtv_median = window.median()
                rtv_skew = window.skew()
                rtv_kurtosis = window.kurtosis()
                rtv_min = window.min()
                rtv_max = window.max()
                rtv_range = rtv_max - rtv_min
                rtv_iqr = np.percentile(window, 75) - np.percentile(window, 25)
            else:
                rtv_mean = np.nan
                rtv_std = np.nan
                rtv_median = np.nan
                rtv_skew = np.nan
                rtv_kurtosis = np.nan
                rtv_min = np.nan
                rtv_max = np.nan
                rtv_range = np.nan
                rtv_iqr = np.nan

            # Assign RTV features to the DataFrame
            actual_idx = group.index[idx]
            df.at[actual_idx, 'rtv_mean'] = rtv_mean
            df.at[actual_idx, 'rtv_std'] = rtv_std
            df.at[actual_idx, 'rtv_median'] = rtv_median
            df.at[actual_idx, 'rtv_skew'] = rtv_skew
            df.at[actual_idx, 'rtv_kurtosis'] = rtv_kurtosis
            df.at[actual_idx, 'rtv_min'] = rtv_min
            df.at[actual_idx, 'rtv_max'] = rtv_max
            df.at[actual_idx, 'rtv_range'] = rtv_range
            df.at[actual_idx, 'rtv_iqr'] = rtv_iqr

    return df

def process_repetition_time_file(gyro_file, body_part, delta_t, repetition_records):
    """
    Process a single gyroscope CSV file to collect repetition durations.

    Parameters:
    - gyro_file (str): Path to the gyroscope CSV file.
    - body_part (str): Name of the body part (e.g., 'Forearm').
    - delta_t (float): Time interval between consecutive samples.
    - repetition_records (list): List to store repetition duration records.

    Returns:
    - None (updates repetition_records in place)
    """
    # Extract Subject ID from the filename
    filename = os.path.basename(gyro_file)
    parts = filename.split('_')
    try:
        subject_id = int(parts[1])
    except (IndexError, ValueError):
        print(f"Filename {filename} does not match the expected format. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(gyro_file)
    except Exception as e:
        print(f"Error reading {gyro_file}: {e}")
        return

    # Verify required columns exist
    required_columns = {'X', 'Y', 'Z', 'Repetition', 'borg'}
    if not required_columns.issubset(df.columns):
        print(f"File {gyro_file} is missing required columns. Skipping.")
        return

    # Group data by Repetition
    repetitions = df['Repetition'].unique()

    for rep in repetitions:
        rep_df = df[df['Repetition'] == rep].reset_index(drop=True)

        # Compute repetition duration
        rep_length = len(rep_df)
        duration = compute_repetition_duration(rep_length, delta_t)

        # Retrieve the Borg scale value (assuming it's constant per repetition)
        borg_value = rep_df['borg'].iloc[0]

        # Append the record
        repetition_records.append({
            'Subject_ID': subject_id,
            'Repetition': rep,
            'borg': borg_value,
            'repetition_duration': duration
        })


def extract_all_repetition_time_variability_features(base_directory=BASE_DIRECTORY,
                                                     body_parts=BODY_PARTS,
                                                     sampling_rate=SAMPLING_RATE,
                                                     output_csv=OUTPUT_RTV_CSV,
                                                     window_size=3):
    """
    Extract repetition time variability features from all gyroscope CSV files and consolidate them into a single CSV.

    Parameters:
    - base_directory (str): Base directory containing the processed data.
    - body_parts (list): List of body parts to process.
    - sampling_rate (int): Sampling rate in Hz.
    - output_csv (str): Path to save the consolidated CSV.
    - window_size (int): Number of repetitions to include in the moving window for RTV.

    Returns:
    - None (saves the consolidated CSV to output_csv)
    """
    delta_t = 1 / sampling_rate

    # Initialize list to store repetition records
    repetition_records = []

    # Iterate through each body part
    for body_part in tqdm(body_parts, desc="Processing Body Parts"):
        gyro_dir = os.path.join(base_directory, body_part, 'gyr')

        if not os.path.isdir(gyro_dir):
            print(f"Gyroscope directory {gyro_dir} does not exist. Skipping body part {body_part}.")
            continue

        # Pattern to match gyroscope CSV files
        file_pattern = os.path.join(gyro_dir, f"Subject_*_{body_part}_gyr_with_borg_gyr.csv")
        gyro_files = glob.glob(file_pattern)

        if not gyro_files:
            print(f"No gyroscope files found in {gyro_dir} for body part {body_part}.")
            continue

        # Process each gyroscope file
        for gyro_file in tqdm(gyro_files, desc=f"Processing {body_part} Gyroscope Files", leave=False):
            process_repetition_time_file(gyro_file, body_part, delta_t, repetition_records)

    # Convert repetition records to DataFrame
    repetition_df = pd.DataFrame(repetition_records)

    if repetition_df.empty:
        print("No repetition records found. RTV features cannot be computed.")
        return

    # Aggregate repetition_duration by Subject_ID and Repetition (average across body parts)
    aggregated_df = repetition_df.groupby(['Subject_ID', 'Repetition', 'borg'], as_index=False).agg(
        {'repetition_duration': 'mean'})

    # Compute RTV features using moving window
    rtv_features_df = extract_repetition_time_variability_features(aggregated_df, window_size=window_size)

    # Save the RTV features DataFrame to CSV
    try:
        rtv_features_df = rtv_features_df.sort_values(by=['Subject_ID', 'Repetition']).reset_index(drop=True)
        rtv_features_df.to_csv(output_csv, index=False)
        print(f"\nConsolidated repetition time variability features saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving consolidated RTV CSV: {e}")


def compute_mpsd(signal, sampling_rate):
    """
    Compute the Maximum Power Spectral Density (MPSD) of a signal using Welch's method.

    Parameters:
    - signal (np.ndarray): 1D array of the signal.
    - sampling_rate (float): Sampling rate in Hz.

    Returns:
    - mpsd (float): Maximum Power Spectral Density value.
    - freqs (np.ndarray): Array of sample frequencies.
    - psd (np.ndarray): Power Spectral Density of the signal.
    """
    # Compute PSD using Welch's method
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=256)

    # Maximum PSD value
    mpsd = np.max(psd)

    return mpsd, freqs, psd


def extract_mpsd_features(sensor_type, sensor_file, body_part, delta_t, feature_dict):
    """
    Extract MPSD features for a given sensor and body part and add them to the feature dictionary.

    Parameters:
    - sensor_type (str): 'acc' or 'gyr'.
    - sensor_file (str): Path to the sensor CSV file.
    - body_part (str): Name of the body part.
    - delta_t (float): Time interval between consecutive samples.
    - feature_dict (dict): Dictionary to store MPSD features.

    Returns:
    - None (updates feature_dict in place)
    """
    # Extract Subject ID from the filename
    filename = os.path.basename(sensor_file)
    parts = filename.split('_')
    try:
        subject_id = int(parts[1])
    except (IndexError, ValueError):
        print(f"Filename {filename} does not match the expected format. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(sensor_file)
    except Exception as e:
        print(f"Error reading {sensor_file}: {e}")
        return

    # Verify required columns exist
    required_columns = {'X', 'Y', 'Z', 'Repetition', 'borg'}
    if not required_columns.issubset(df.columns):
        print(f"File {sensor_file} is missing required columns. Skipping.")
        return

    # Fill missing borg values with the maximum available value
    if df['borg'].isnull().any():
        max_borg = df['borg'].max()
        if not np.isnan(max_borg):
            df['borg'].fillna(max_borg, inplace=True)
        else:
            print(f"Missing 'borg' values in {sensor_file}, and no valid max value found. Assigning NaN.")
            df['borg'].fillna(np.nan, inplace=True)

    # Group data by Repetition
    repetitions = df['Repetition'].unique()

    for rep in repetitions:
        rep_df = df[df['Repetition'] == rep].reset_index(drop=True)

        # Retrieve the Borg scale value
        if rep_df.empty or 'borg' not in rep_df.columns:
            borg_value = np.nan
            print(f"Missing 'borg' value for Subject {subject_id}, Repetition {rep}. Assigning NaN.")
        else:
            borg_value = rep_df['borg'].iloc[0]

        # Initialize a unique key for each Subject and Repetition
        key = (subject_id, rep, borg_value)

        # Always add the repetition to the feature dictionary
        if key not in feature_dict:
            feature_dict[key] = {'Subject_ID': subject_id, 'Repetition': rep, 'borg': borg_value}

        # Extract axis data
        axis_x = rep_df['X'].values if 'X' in rep_df else np.array([])
        axis_y = rep_df['Y'].values if 'Y' in rep_df else np.array([])
        axis_z = rep_df['Z'].values if 'Z' in rep_df else np.array([])

        # Initialize MPSD values as NaN
        mpsd_x, mpsd_y, mpsd_z, mpsd_combined = np.nan, np.nan, np.nan, np.nan

        # Compute MPSD only if there are enough data points
        if len(axis_x) >= 128:
            mpsd_x, _, _ = compute_mpsd(axis_x, SAMPLING_RATE)
            mpsd_y, _, _ = compute_mpsd(axis_y, SAMPLING_RATE)
            mpsd_z, _, _ = compute_mpsd(axis_z, SAMPLING_RATE)
            mpsd_combined = max(mpsd_x, mpsd_y, mpsd_z)
        else:
            print(f"Not enough data points in Subject {subject_id}, Repetition {rep}, Body Part {body_part}, Sensor {sensor_type}. Assigning NaN values for MPSD.")

        # Assign MPSD values (or NaN) to the feature dictionary
        feature_dict[key][f"mpsd_{sensor_type}_{body_part.lower()}_x"] = mpsd_x
        feature_dict[key][f"mpsd_{sensor_type}_{body_part.lower()}_y"] = mpsd_y
        feature_dict[key][f"mpsd_{sensor_type}_{body_part.lower()}_z"] = mpsd_z
        feature_dict[key][f"mpsd_{sensor_type}_{body_part.lower()}_combined"] = mpsd_combined




def extract_all_mpsd_features(base_directory=BASE_DIRECTORY,
                              body_parts=BODY_PARTS,
                              sampling_rate=SAMPLING_RATE,
                              output_csv=OUTPUT_MPSD_CSV):
    """
    Extract MPSD features from all accelerometer and gyroscope CSV files and consolidate them into a single CSV.

    Parameters:
    - base_directory (str): Base directory containing the processed data.
    - body_parts (list): List of body parts to process.
    - sampling_rate (int): Sampling rate in Hz.
    - output_csv (str): Path to save the consolidated CSV.

    Returns:
    - None (saves the consolidated CSV to output_csv)
    """
    delta_t = 1 / sampling_rate

    # Initialize feature dictionary
    feature_dict = {}

    # Iterate through each body part
    for body_part in tqdm(body_parts, desc="Processing Body Parts"):
        for sensor_type in ['acc', 'gyr']:
            sensor_dir = os.path.join(base_directory, body_part, sensor_type)

            if not os.path.isdir(sensor_dir):
                print(
                    f"{sensor_type.upper()} directory {sensor_dir} does not exist. Skipping Body Part {body_part}, Sensor {sensor_type}.")
                continue

            # Pattern to match sensor CSV files
            file_pattern = os.path.join(sensor_dir, f"Subject_*_{body_part}_{sensor_type}_with_borg_{sensor_type}.csv")
            sensor_files = glob.glob(file_pattern)

            if not sensor_files:
                print(f"No {sensor_type.upper()} files found in {sensor_dir} for Body Part {body_part}.")
                continue

            # Process each sensor file
            for sensor_file in tqdm(sensor_files, desc=f"Processing {sensor_type.upper()} Files for {body_part}",
                                    leave=False):
                extract_mpsd_features(sensor_type, sensor_file, body_part, delta_t, feature_dict)

    # Convert feature dictionary to DataFrame
    mpsd_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # Reset index to have Subject_ID and Repetition as columns
    mpsd_df.reset_index(drop=True, inplace=True)

    # Save to CSV
    try:
        mpsd_df = mpsd_df.sort_values(by=['Subject_ID', 'Repetition']).reset_index(drop=True)
        mpsd_df.to_csv(output_csv, index=False)
        print(f"MPSD features saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving MPSD CSV: {e}")


def compute_angular_displacement(angular_velocity, delta_t):
    """
    Compute angular displacement by integrating angular velocity using the trapezoidal rule.

    Parameters:
    - angular_velocity (np.ndarray): Angular velocity data.
    - delta_t (float): Time interval between consecutive samples.

    Returns:
    - angular_displacement (np.ndarray): Computed angular displacement.
    """
    # Numerically integrate angular velocity to get angular displacement
    # cumtrapz returns an array of length N-1; prepend 0 to align lengths
    angular_displacement = np.concatenate(([0], cumulative_trapezoid(angular_velocity, dx=delta_t)))
    return angular_displacement


def extract_rom_features(angular_displacement_x, angular_displacement_y, angular_displacement_z):
    """
    Extract ROM features for specified axes and compute combined ROM magnitude.

    Parameters:
    - angular_displacement_x (np.ndarray): Angular displacement along X-axis.
    - angular_displacement_y (np.ndarray): Angular displacement along Y-axis.
    - angular_displacement_z (np.ndarray): Angular displacement along Z-axis.

    Returns:
    - rom_features (dict): Dictionary containing ROM features for each axis and combined.
    """
    # Calculate ROM for each axis
    rom_x = np.max(angular_displacement_x) - np.min(angular_displacement_x)
    rom_y = np.max(angular_displacement_y) - np.min(angular_displacement_y)
    rom_z = np.max(angular_displacement_z) - np.min(angular_displacement_z)

    # Compute combined ROM magnitude
    rom_combined = np.sqrt(rom_x ** 2 + rom_y ** 2 + rom_z ** 2)

    return {
        'rom_x': rom_x,
        'rom_y': rom_y,
        'rom_z': rom_z,
        'rom_combined': rom_combined
    }


def extract_rom_features_from_gyro(gyro_x, gyro_y, gyro_z, delta_t, body_part):
    """
    Extract ROM features from gyroscope data for a specific body part.

    Parameters:
    - gyro_x (np.ndarray): Gyroscope data along X-axis.
    - gyro_y (np.ndarray): Gyroscope data along Y-axis.
    - gyro_z (np.ndarray): Gyroscope data along Z-axis.
    - delta_t (float): Time interval between consecutive samples.
    - body_part (str): Name of the body part.

    Returns:
    - rom_features (dict): Dictionary containing ROM features with appropriate naming.
    """
    # Compute angular displacement for each axis
    angular_displacement_x = compute_angular_displacement(gyro_x, delta_t)
    angular_displacement_y = compute_angular_displacement(gyro_y, delta_t)
    angular_displacement_z = compute_angular_displacement(gyro_z, delta_t)

    # Extract ROM features
    rom = extract_rom_features(angular_displacement_x, angular_displacement_y, angular_displacement_z)

    # Define feature names based on body part and axis
    rom_features = {}
    relevant_axes = get_relevant_axes(body_part)

    for axis in ['x', 'y', 'z']:
        displacement = {
            'x': angular_displacement_x,
            'y': angular_displacement_y,
            'z': angular_displacement_z
        }[axis]

        rom_axis = rom[f'rom_{axis}']
        rom_combined = rom['rom_combined']
        rom_feature_name = f"rom_{body_part.lower()}_{axis}"
        rom_combined_name = f"rom_{body_part.lower()}_combined"
        rom_features[rom_feature_name] = rom_axis
        rom_features[rom_combined_name] = rom_combined
        rom_features[f'rom_{body_part.lower()}_{axis}_mean'] = np.mean(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_median'] = np.median(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_std'] = np.std(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_skew'] = skew(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_kurtosis'] = kurtosis(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_min'] = np.min(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_max'] = np.max(displacement)
        rom_features[f'rom_{body_part.lower()}_{axis}_range'] = rom[f'rom_{axis}']
        rom_features[f'rom_{body_part.lower()}_{axis}_iqr'] = np.percentile(displacement, 75) - np.percentile(displacement, 25)

    displacement_combined = np.sqrt(
        angular_displacement_x ** 2 + angular_displacement_y ** 2 + angular_displacement_z ** 2)
    rom_features[f'rom_{body_part.lower()}_combined_mean'] = np.mean(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_median'] = np.median(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_std'] = np.std(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_skew'] = skew(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_kurtosis'] = kurtosis(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_min'] = np.min(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_max'] = np.max(displacement_combined)
    rom_features[f'rom_{body_part.lower()}_combined_range'] = rom['rom_combined']
    rom_features[f'rom_{body_part.lower()}_combined_iqr'] = np.percentile(displacement_combined, 75) - np.percentile(
        displacement_combined, 25)

    # for axis in ['x', 'y', 'z']:
    #     if axis in relevant_axes:
    #         rom_axis = rom[f'rom_{axis}']
    #         rom_combined = rom['rom_combined']
    #         rom_feature_name = f"rom_{body_part.lower()}_{axis}"
    #         rom_combined_name = f"rom_{body_part.lower()}_combined"
    #         rom_features[rom_feature_name] = rom_axis
    #         rom_features[rom_combined_name] = rom_combined
    #     else:
    #         # If axis is not relevant for the body part, set as NaN
    #         rom_feature_name = f"rom_{body_part.lower()}_{axis}"
    #         rom_combined_name = f"rom_{body_part.lower()}_combined"
    #         rom_features[rom_feature_name] = np.nan
    #         rom_features[rom_combined_name] = np.nan

    return rom_features


def get_relevant_axes(body_part):
    """
    Get the relevant axis or axes for ROM computation based on the body part.

    Parameters:
    - body_part (str): Name of the body part.

    Returns:
    - axes (list): List of relevant axes ('x', 'y', 'z').
    """
    relevant_axes_map = {
        'Forearm': ['y'],
        'Shoulder': ['y'],
        'Palm': ['y'],
        'Torso': ['x'],
        'Pelvis': ['x']
    }
    return relevant_axes_map.get(body_part, [])


def extract_rom_features_from_gyro_file(gyro_file, body_part, delta_t, feature_dict):
    """
    Extract ROM features from a single gyroscope CSV file and add them to the feature dictionary.

    Parameters:
    - gyro_file (str): Path to the gyroscope CSV file.
    - body_part (str): Name of the body part.
    - delta_t (float): Time interval between consecutive samples.
    - feature_dict (dict): Dictionary to store ROM features.

    Returns:
    - None (updates feature_dict in place)
    """
    # Extract Subject ID from the filename
    filename = os.path.basename(gyro_file)
    parts = filename.split('_')
    try:
        subject_id = int(parts[1])
    except (IndexError, ValueError):
        print(f"Filename {filename} does not match the expected format. Skipping.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(gyro_file)
    except Exception as e:
        print(f"Error reading {gyro_file}: {e}")
        return

    # Verify required columns exist
    required_columns = {'X', 'Y', 'Z', 'Repetition', 'borg'}
    if not required_columns.issubset(df.columns):
        print(f"File {gyro_file} is missing required columns. Skipping.")
        return

    # Group data by Repetition
    repetitions = df['Repetition'].unique()

    for rep in repetitions:
        rep_df = df[df['Repetition'] == rep].reset_index(drop=True)

        # Extract X, Y, Z gyroscope data
        gyro_x = rep_df['X'].values
        gyro_y = rep_df['Y'].values
        gyro_z = rep_df['Z'].values

        # Check if there's enough data to compute ROM
        if len(gyro_x) < 2:
            print(
                f"Not enough data points in Subject {subject_id}, Repetition {rep} for body part {body_part}. Skipping ROM features for this repetition.")
            continue

        # Extract ROM features
        rom_features = extract_rom_features_from_gyro(gyro_x, gyro_y, gyro_z, delta_t, body_part)

        # Retrieve the Borg scale value (assuming it's constant per repetition)
        borg_value = rep_df['borg'].iloc[0]

        # Initialize a unique key for each Subject and Repetition
        key = (subject_id, rep)

        if key not in feature_dict:
            feature_dict[key] = {'Subject_ID': subject_id, 'Repetition': rep, 'borg': borg_value}
        else:
            # Verify Borg consistency
            if not np.isclose(feature_dict[key]['borg'], borg_value):
                print(
                    f"Warning: Inconsistent Borg values for Subject {subject_id}, Repetition {rep}. Using the first encountered value.")

        # Assign the computed ROM features
        for feature_name, value in rom_features.items():
            feature_dict[key][feature_name] = value


def extract_all_rom_features(base_directory=BASE_DIRECTORY,
                             body_parts=BODY_PARTS,
                             sampling_rate=SAMPLING_RATE,
                             output_csv=OUTPUT_ROM_CSV):
    """
    Extract ROM features from all gyroscope CSV files and consolidate them into a single CSV.

    Parameters:
    - base_directory (str): Base directory containing the processed data.
    - body_parts (list): List of body parts to process.
    - sampling_rate (int): Sampling rate in Hz.
    - output_csv (str): Path to save the consolidated CSV.

    Returns:
    - None (saves the consolidated CSV to output_csv)
    """
    delta_t = 1 / sampling_rate

    # Initialize feature dictionary
    feature_dict = {}

    # Iterate through each body part
    for body_part in tqdm(body_parts, desc="Processing Body Parts for ROM"):
        gyro_dir = os.path.join(base_directory, body_part, 'gyr')

        if not os.path.isdir(gyro_dir):
            print(f"Gyroscope directory {gyro_dir} does not exist. Skipping body part {body_part}.")
            continue

        # Pattern to match gyroscope CSV files
        file_pattern = os.path.join(gyro_dir, f"Subject_*_{body_part}_gyr_with_borg_gyr.csv")
        gyro_files = glob.glob(file_pattern)

        if not gyro_files:
            print(f"No gyroscope files found in {gyro_dir} for body part {body_part}.")
            continue

        # Process each gyroscope file
        for gyro_file in tqdm(gyro_files, desc=f"Processing {body_part} Gyroscope Files for ROM", leave=False):
            extract_rom_features_from_gyro_file(gyro_file, body_part, delta_t, feature_dict)

    # Convert feature dictionary to DataFrame
    rom_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # Reset index to have Subject_ID and Repetition as columns
    rom_df.reset_index(drop=True, inplace=True)

    # Save to CSV
    try:
        rom_df = rom_df.sort_values(by=['Subject_ID', 'Repetition']).reset_index(drop=True)
        rom_df.to_csv(output_csv, index=False)
        print(f"ROM features saved to '{output_csv}'")
    except Exception as e:
        print(f"Error saving ROM CSV: {e}")


def compute_additional_features(merged_csv=MERGED_CSV):
    """
    Compute additional features based on Repetition and existing features.

    Features to compute:
    1. Normalized Repetition-Based Features
        - Normalized_Repetition = Repetition / Max(Repetition for Subject)
        - Normalized Jerk = Jerk * Normalized Repetition
        - Normalized ROM = ROM * Normalized Repetition
    2. Cumulative Features
        - Cumulative Jerk: Cumulative sum of Jerk values up to current repetition.
        - Cumulative ROM: Cumulative sum of ROM values up to current repetition.
    3. Rate of Change Features
        - Rate of Change in Jerk = (Jerk at Rep i - Jerk at Rep i-1) / Repetition
        - Rate of Change in ROM = (ROM at Rep i - ROM at Rep i-1) / Repetition
    4. Interaction Features
        - Repetition × Jerk = Jerk * Repetition
        - Repetition × ROM = ROM * Repetition
    5. Trend Features
        - Slope of Jerk vs. Repetition
        - Slope of ROM vs. Repetition

    All features are computed **per subject** to ensure relevance.

    Parameters:
    - merged_csv (str): Path to the merged CSV file.

    Returns:
    - None (saves updated DataFrame back to merged_csv)
    """
    import scipy.stats as stats

    # Read the merged CSV
    df = pd.read_csv(merged_csv)

    # Ensure that Repetition is numeric
    df['Repetition'] = pd.to_numeric(df['Repetition'], errors='coerce')

    # Identify Jerk and ROM feature columns
    # Assuming Jerk features start with 'jerk_' and ROM features contain 'rom_'
    # Further assuming ROM combined features contain 'combined' in their column names
    jerk_features = [col for col in df.columns if col.startswith('jerk_')]
    rom_features = [col for col in df.columns if col.startswith('rom_') and 'combined' in col]

    # Initialize new feature columns
    # 1. Normalized Repetition-Based Features
    df['Normalized_Repetition'] = np.nan
    for jerk_col in jerk_features:
        df[f'Normalized_{jerk_col}'] = np.nan
    for rom_col in rom_features:
        df[f'Normalized_{rom_col}'] = np.nan

    # 2. Cumulative Features
    for jerk_col in jerk_features:
        df[f'Cumulative_{jerk_col}'] = np.nan
    for rom_col in rom_features:
        df[f'Cumulative_{rom_col}'] = np.nan

    # 3. Rate of Change Features
    for jerk_col in jerk_features:
        df[f'Rate_of_Change_{jerk_col}'] = np.nan
    for rom_col in rom_features:
        df[f'Rate_of_Change_{rom_col}'] = np.nan

    # 4. Interaction Features
    for jerk_col in jerk_features:
        df[f'Interaction_{jerk_col}_Repetition'] = np.nan
    for rom_col in rom_features:
        df[f'Interaction_{rom_col}_Repetition'] = np.nan

    # 5. Trend Features
    for jerk_col in jerk_features:
        df[f'Trend_Slope_{jerk_col}_vs_Repetition'] = np.nan
    for rom_col in rom_features:
        df[f'Trend_Slope_{rom_col}_vs_Repetition'] = np.nan

    # Group by Subject_ID
    grouped = df.groupby('Subject_ID')

    for subject_id, group in tqdm(grouped, desc="Computing Additional Features"):
        # Sort by Repetition
        group = group.sort_values('Repetition').reset_index()
        indices = group['index'].values
        repetitions = group['Repetition'].values

        if len(repetitions) == 0:
            continue

        max_repetition = repetitions.max()
        if max_repetition == 0:
            normalized_repetition = np.nan * np.ones_like(repetitions)
        else:
            normalized_repetition = repetitions / max_repetition

        # Assign Normalized Repetition
        df.loc[indices, 'Normalized_Repetition'] = normalized_repetition

        # Compute Normalized Jerk Features
        for jerk_col in jerk_features:
            normalized_col = f'Normalized_{jerk_col}'
            df.loc[indices, normalized_col] = df.loc[indices, jerk_col] * normalized_repetition

        # Compute Normalized ROM Features
        for rom_col in rom_features:
            normalized_col = f'Normalized_{rom_col}'
            df.loc[indices, normalized_col] = df.loc[indices, rom_col] * normalized_repetition

        # Compute Cumulative Jerk Features
        for jerk_col in jerk_features:
            cumulative_col = f'Cumulative_{jerk_col}'
            df.loc[indices, cumulative_col] = df.loc[indices, jerk_col].cumsum()

        # Compute Cumulative ROM Features
        for rom_col in rom_features:
            cumulative_col = f'Cumulative_{rom_col}'
            df.loc[indices, cumulative_col] = df.loc[indices, rom_col].cumsum()

        # Compute Rate of Change Features
        for jerk_col in jerk_features:
            roc_col = f'Rate_of_Change_{jerk_col}'
            # Calculate difference with previous repetition
            jerk_diff = df.loc[indices, jerk_col].diff()
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                roc = jerk_diff / repetitions
            df.loc[indices, roc_col] = roc

        for rom_col in rom_features:
            roc_col = f'Rate_of_Change_{rom_col}'
            rom_diff = df.loc[indices, rom_col].diff()
            with np.errstate(divide='ignore', invalid='ignore'):
                roc = rom_diff / repetitions
            df.loc[indices, roc_col] = roc

        # Compute Interaction Features
        for jerk_col in jerk_features:
            interaction_col = f'Interaction_{jerk_col}_Repetition'
            df.loc[indices, interaction_col] = df.loc[indices, jerk_col] * repetitions

        for rom_col in rom_features:
            interaction_col = f'Interaction_{rom_col}_Repetition'
            df.loc[indices, interaction_col] = df.loc[indices, rom_col] * repetitions

        # Compute Trend Features (Slope of Jerk vs Repetition and ROM vs Repetition)
        for jerk_col in jerk_features:
            trend_col = f'Trend_Slope_{jerk_col}_vs_Repetition'
            # Check if there are at least two repetitions with non-NaN Jerk
            valid_indices = ~df.loc[indices, jerk_col].isna()
            if valid_indices.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(repetitions[valid_indices],
                                                                         df.loc[indices, jerk_col][valid_indices])
                df.loc[indices, trend_col] = slope
            else:
                df.loc[indices, trend_col] = np.nan

        for rom_col in rom_features:
            trend_col = f'Trend_Slope_{rom_col}_vs_Repetition'
            # Check if there are at least two repetitions with non-NaN ROM
            valid_indices = ~df.loc[indices, rom_col].isna()
            if valid_indices.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(repetitions[valid_indices],
                                                                         df.loc[indices, rom_col][valid_indices])
                df.loc[indices, trend_col] = slope
            else:
                df.loc[indices, trend_col] = np.nan

    # Save back to merged.csv
    try:
        df.to_csv(merged_csv, index=False)
        print(f"Additional features saved to '{merged_csv}'.")
    except Exception as e:
        print(f"Error saving additional features: {e}")


def merge_features(jerk_csv=OUTPUT_JERK_CSV,
                   freq_csv=OUTPUT_FREQ_CSV,
                   mpsd_csv=OUTPUT_MPSD_CSV,
                   rom_csv=OUTPUT_ROM_CSV,
                   merged_csv="OutputCSVFiles/Merged.csv"):
    """
    Merge jerk, movement frequency, and MPSD features into a single CSV.

    Parameters:
    - jerk_csv (str): Path to the jerk features CSV.
    - freq_csv (str): Path to the movement frequency features CSV.
    - mpsd_csv (str): Path to the MPSD features CSV.
    - merged_csv (str): Path to save the merged CSV.

    Returns:
    - None (saves the merged CSV to merged_csv)
    """
    try:
        # Read jerk features
        jerk_df = pd.read_csv(jerk_csv)
        print(f"Jerk features loaded from '{jerk_csv}'.")

        # Read movement frequency features
        freq_df = pd.read_csv(freq_csv)
        print(f"Movement Frequency features loaded from '{freq_csv}'.")

        # Read MPSD features
        mpsd_df = pd.read_csv(mpsd_csv)
        print(f"MPSD features loaded from '{mpsd_csv}'.")

        # Read ROM features
        rom_df = pd.read_csv(rom_csv)
        print(f"ROM features loaded from '{rom_csv}'.")

        # Merge all feature files in a single statement
        merged_df = (
            jerk_df
                .merge(freq_df, on=['Subject_ID', 'Repetition', 'borg'], how='outer')
                .merge(mpsd_df, on=['Subject_ID', 'Repetition', 'borg'], how='outer')
                .merge(rom_df, on=['Subject_ID', 'Repetition', 'borg'], how='outer')
        )

        # Save the merged DataFrame
        merged_df.to_csv(merged_csv, index=False)
        print(f"Merged features saved to '{merged_csv}'.")
    except Exception as e:
        print(f"Error merging features: {e}")



# Path to the raw segmented files
raw_data_dir = "processed_data_35_i"

# Function to read raw segmented files and calculate repetitions
def count_raw_repetitions(base_dir):
    raw_reps_count = {}

    # Iterate through all body part directories
    for body_part in os.listdir(base_dir):
        body_part_dir = os.path.join(base_dir, body_part)
        if not os.path.isdir(body_part_dir):
            continue

        # Iterate through sensor subdirectories (e.g., acc, gyr)
        for sensor_type in os.listdir(body_part_dir):
            sensor_dir = os.path.join(body_part_dir, sensor_type)
            if not os.path.isdir(sensor_dir):
                continue

            # Iterate through CSV files in the sensor directory
            for csv_file in os.listdir(sensor_dir):
                if csv_file.endswith(".csv"):
                    # Extract subject ID from the filename
                    subject_id = int(csv_file.split('_')[1])  # Assuming filename format: Subject_<ID>_...
                    file_path = os.path.join(sensor_dir, csv_file)

                    # Read the CSV file
                    try:
                        df = pd.read_csv(file_path)
                        # Count unique repetitions for the subject
                        repetitions = df['Repetition'].nunique()

                        # Update the dictionary with max repetitions per subject
                        if subject_id not in raw_reps_count:
                            raw_reps_count[subject_id] = repetitions
                        else:
                            raw_reps_count[subject_id] = max(raw_reps_count[subject_id], repetitions)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    return raw_reps_count


def compare_number_of_repititions():
    # Read raw data and calculate repetitions
    raw_reps_count = count_raw_repetitions(raw_data_dir)
    raw_reps_df = pd.DataFrame(list(raw_reps_count.items()), columns=['Subject_ID', 'Raw_Repetitions']).set_index(
        'Subject_ID')

    # Read feature files
    mpsd_df = pd.read_csv('OutputCSVFiles/consolidated_mpsd_features.csv')
    jerk_df = pd.read_csv('OutputCSVFiles/consolidated_jerk_features.csv')
    movfreq_df = pd.read_csv('OutputCSVFiles/consolidated_movFreq_features.csv')

    # Calculate repetitions in feature files
    mpsd_reps_count = mpsd_df.groupby('Subject_ID')['Repetition'].nunique()
    jerk_reps_count = jerk_df.groupby('Subject_ID')['Repetition'].nunique()
    movfreq_reps_count = movfreq_df.groupby('Subject_ID')['Repetition'].nunique()

    # Combine all counts into a single DataFrame
    reps_comparison = pd.DataFrame({
        'Raw_Repetitions': raw_reps_df['Raw_Repetitions'],
        'MPSD': mpsd_reps_count,
        'Jerk': jerk_reps_count,
        'Movement_Frequency': movfreq_reps_count
    }).fillna(0).astype(int)

    # Print the comparison
    print("\nComparison of repetitions across raw data and feature files:")
    print(reps_comparison)