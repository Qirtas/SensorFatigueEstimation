import pandas as pd
import numpy as np
import os


def compute_repetition_times_for_all_subjects(folder_path, sampling_rate=100, start_offset=12.0):

    """
    Compute start, end, and midpoint times for all subject files in the given folder.

    Parameters:
    ----------
    folder_path : str
        Path to the folder containing segmented IMU CSV files for all subjects.
    sampling_rate : int
        Sampling frequency in Hz. Default is 100 Hz.
    start_offset : float
        Offset for the start time in seconds (e.g., 8 seconds excluded). Default is 8.0 seconds.

    Returns:
    -------
    all_repetition_times : pd.DataFrame
        DataFrame with Subject, Repetition, Start Time, End Time, and Midpoint columns.
    """

    # Sampling period (seconds per sample)
    sampling_period = 1 / sampling_rate

    all_repetition_times = []

    # Iterate through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            subject_id = file_name.split("_")[1]  # Extract subject ID from the file name
            file_path = os.path.join(folder_path, file_name)

            # Load the IMU segmented data
            data = pd.read_csv(file_path)

            # Initialize start offset for this subject
            subject_start_offset = start_offset

            # Group by Repetition to compute times
            for repetition, group in data.groupby("Repetition"):
                # Number of samples in the repetition
                num_samples = len(group)

                # Compute start, end, and midpoint times
                start_time = subject_start_offset
                end_time = start_time + num_samples * sampling_period
                midpoint = (start_time + end_time) / 2

                # Append the repetition times for this subject
                all_repetition_times.append({
                    "Subject": subject_id,
                    "Repetition": repetition,
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Midpoint": midpoint
                })

                # Update the start_offset for the next repetition
                subject_start_offset = end_time

    # Convert to a DataFrame
    return pd.DataFrame(all_repetition_times)


# Using Interpolation Approach
def interpolate_borg_values(repetition_file, borg_file, output_file):
    """
    Map Borg scale values to repetitions using linear interpolation.

    Parameters:
    ----------
    repetition_file : str
        Path to the CSV file containing repetition times with columns:
        'Subject', 'Repetition', 'Start Time', 'End Time', 'Midpoint'
    borg_file : str
        Path to the CSV file containing Borg ratings with time intervals
    output_file : str
        Path to save the output CSV with interpolated Borg values.

    Returns:
    -------
    None
    """
    # Load repetition times and Borg data
    repetition_data = pd.read_csv(repetition_file)
    borg_data = pd.read_csv(borg_file)

    # Convert Borg data from wide to long format
    # Assuming your Borg data is in the format you showed earlier
    borg_long = pd.melt(
        borg_data,
        id_vars=['subject', 'task_order', 'before_task'],
        value_vars=[col for col in borg_data.columns if 'sec' in col],
        var_name='time_point',
        value_name='Borg'
    )

    # Convert time points to seconds
    borg_long['time_seconds'] = borg_long['time_point'].str.extract('(\d+)').astype(float)

    # Sort by subject and time
    borg_long = borg_long.sort_values(['subject', 'time_seconds'])

    # Process each subject separately
    all_interpolated_data = []

    for subject in repetition_data['Subject'].unique():
        # Get repetition data for this subject
        subject_repetitions = repetition_data[repetition_data['Subject'] == subject].copy()

        # Get Borg data for this subject
        subject_borg = borg_long[borg_long['subject'] == f'subject_{subject}']

        if len(subject_borg) > 0:  # Check if we have Borg data for this subject
            # Interpolate Borg values for this subject's repetitions
            subject_repetitions['Borg'] = np.interp(
                subject_repetitions['Midpoint'],
                subject_borg['time_seconds'],
                subject_borg['Borg'],
                left=subject_borg['Borg'].iloc[0],  # Use first Borg value for times before first measurement
                right=subject_borg['Borg'].iloc[-1]  # Use last Borg value for times after last measurement
            )

            # Exclude repetitions beyond the final Borg recording time + 10 seconds
            max_borg_time = subject_borg['time_seconds'].max()
            subject_repetitions = subject_repetitions[
                subject_repetitions['Midpoint'] <= max_borg_time + 10
            ]

            all_interpolated_data.append(subject_repetitions)

    # Combine all subjects' data
    if all_interpolated_data:
        final_data = pd.concat(all_interpolated_data, ignore_index=True)

        # Save the output
        final_data.to_csv(output_file, index=False)
        print(f"Interpolated Borg values saved to {output_file}.")
    else:
        print("No data to save. Please check if the subject IDs match between the files.")




# Using Midpoint Times
def map_borg_to_repetitions(repetition_file, borg_file, output_file):
    """
    Map Borg scale values to repetition data based on midpoint times.

    Parameters:
    ----------
    repetition_file : str
        Path to the CSV file containing repetition times.
    borg_file : str
        Path to the CSV file containing Borg ratings.
    output_file : str
        Path to save the output CSV with mapped Borg values.

    Returns:
    -------
    None
    """
    # Load files
    repetition_data = pd.read_csv(repetition_file)
    borg_data = pd.read_csv(borg_file)

    # Melt Borg data to long format for easier mapping
    borg_long = borg_data.melt(
        id_vars=["subject"],
        var_name="Time_Interval",
        value_name="Borg"
    )

    # Extract numeric seconds from the time interval column (e.g., "10_sec" -> 10)
    borg_long["Time_Seconds"] = borg_long["Time_Interval"].str.extract(r"(\d+)_sec").astype(float)

    # Merge Borg values based on subject and time intervals
    mapped_data = []
    for _, row in repetition_data.iterrows():
        subject = row["Subject"]
        midpoint = row["Midpoint"]

        # Filter Borg data for the same subject
        subject_borg = borg_long[borg_long["subject"] == f"subject_{int(subject)}"]

        # Find the closest Borg interval for the midpoint
        closest_borg = subject_borg.iloc[(subject_borg["Time_Seconds"] - midpoint).abs().argmin()]
        borg_value = closest_borg["Borg"]

        # Handle missing Borg values
        # if pd.isnull(borg_value):
        #     borg_value = 15

        #
        # if borg_value in [18, 19]:
        #     if np.random.rand() < 0.7:  # 70% chance to reassign
        #         # Probabilistic reassignment to 15 or 16
        #         if np.random.rand() < 0.6:  # 60% chance to assign 16
        #             borg_value = 16
        #         else:  # 40% chance to assign 15
        #             borg_value = 15

        # Append the Borg value to the repetition row
        row["Borg"] = borg_value
        mapped_data.append(row)

    # Convert to DataFrame and save
    final_data = pd.DataFrame(mapped_data)
    final_data.to_csv(output_file, index=False)
    print(f"Mapped Borg values saved to {output_file}.")





'''
def map_borg_to_repetitions(
    repetition_file,
    borg_file,
    output_file,
    max_high_borg_fraction=0.15,
    lower_reassign_range=(14, 16),
    keep_high_for_last_fraction=0.20
):
    """
    Map Borg scale values to repetition data based on midpoint times.
    Adds robust reassignment logic for Borg = 18 or 19, and
    enforces a strictly ascending (or stable) Borg progression
    as repetition indices increase.

    Parameters:
    -----------
    repetition_file : str
        Path to the CSV file containing repetition times.
        Must have columns: ['Subject', 'Midpoint', ...].
    borg_file : str
        Path to the CSV file containing Borg ratings.
        Must have columns: ["subject", ...] for Borg values at various times.
    output_file : str
        Path to save the output CSV with mapped Borg values.
    max_high_borg_fraction : float
        Fraction of total repetitions allowed to remain at Borg 18 or 19.
    lower_reassign_range : (int, int)
        Range from which we draw replacement values if we reduce 18/19
        (e.g., 14–16).
    keep_high_for_last_fraction : float
        Fraction of final repetitions (per participant) that we allow to keep 18/19
        if originally assigned.

    Returns:
    --------
    None
    """

    # --- Step 1: Load files ---
    repetition_data = pd.read_csv(repetition_file)
    borg_data = pd.read_csv(borg_file)

    # Melt Borg data to long format for easier mapping
    borg_long = borg_data.melt(
        id_vars=["subject"],
        var_name="Time_Interval",
        value_name="Borg"
    )

    # Extract numeric seconds from the time interval column (e.g., "10_sec" -> 10)
    borg_long["Time_Seconds"] = borg_long["Time_Interval"].str.extract(r"(\d+)_sec").astype(float)

    # Dictionary to store subject-specific repetition data
    subject_reps = {}

    # --- Step 2: For each repetition row, find the closest Borg value ---
    for _, row in repetition_data.iterrows():
        subject = row["Subject"]
        midpoint = row["Midpoint"]

        # Filter Borg data for the same subject
        subject_key = f"subject_{int(subject)}"
        subject_borg = borg_long[borg_long["subject"] == subject_key]

        if subject_borg.empty:
            # If we have no data for this subject, default Borg=19
            row["Borg"] = 19
        else:
            # Find the closest Borg interval for the midpoint
            closest_borg = subject_borg.iloc[
                (subject_borg["Time_Seconds"] - midpoint).abs().argmin()
            ]
            borg_value = closest_borg["Borg"]

            # Handle missing Borg => default to 19
            if pd.isnull(borg_value):
                borg_value = 19

            row["Borg"] = borg_value

        # Accumulate in a dictionary keyed by subject
        subject_reps.setdefault(subject, []).append(row)

    # --- Step 3: Process each subject's data for 18/19 reassignments ---
    final_rows = []

    for subj, rows in subject_reps.items():
        # Convert to DataFrame for manipulation
        df_subj = pd.DataFrame(rows)

        # Sort by 'Midpoint' (chronological order)
        df_subj.sort_values(by="Midpoint", inplace=True)
        df_subj.reset_index(drop=True, inplace=True)

        total_reps = len(df_subj)
        if total_reps == 0:
            continue

        high_borg_mask = df_subj["Borg"].isin([18, 19])
        high_count = high_borg_mask.sum()

        # Number of 18/19 we want to keep
        allowed_high_count = int(np.floor(max_high_borg_fraction * total_reps))

        # Number of reps in the last fraction (where 18/19 can remain)
        last_fraction_count = int(np.ceil(keep_high_for_last_fraction * total_reps))
        cutoff_idx = total_reps - last_fraction_count

        # We track how many high Borg we've allowed so far
        high_assigned = 0

        for i in range(total_reps):
            original_borg = df_subj.loc[i, "Borg"]
            if original_borg in [18, 19]:
                # If this repetition is before the last fraction or we already hit the limit
                if i < cutoff_idx or (high_assigned >= allowed_high_count):
                    # Reassign to a lower range (14–16)
                    df_subj.loc[i, "Borg"] = np.random.randint(
                        lower_reassign_range[0], lower_reassign_range[1] + 1
                    )
                else:
                    # Keep it
                    high_assigned += 1

        # --- Step 4: Enforce strictly ascending (or stable) progression ---
        # We'll do a single forward pass to ensure that each repetition's Borg >= the previous one.
        # This ensures no dips back down to 14 after a 16, for example.

        borg_values = df_subj["Borg"].values
        for i in range(1, total_reps):
            if borg_values[i] < borg_values[i - 1]:
                borg_values[i] = borg_values[i - 1]
            # Also cap at 19 if we exceed
            if borg_values[i] > 19:
                borg_values[i] = 19

        df_subj["Borg"] = borg_values

        final_rows.append(df_subj)

    # Concatenate all processed subjects
    final_data = pd.concat(final_rows, axis=0)
    final_data.reset_index(drop=True, inplace=True)

    # --- Step 5: Save to CSV ---
    final_data.to_csv(output_file, index=False)
    print(f"Reassigned and sorted Borg values saved to {output_file}.")
'''



def add_borg_to_features(features_file, repetition_file, output_file):
    """
    Add the Borg column from repetition times file to the jerk features file.

    Parameters:
    ----------
    features_file : str
        Path to the CSV file containing jerk features.
    repetition_file : str
        Path to the CSV file containing repetition times with Borg values.
    output_file : str
        Path to save the updated features file with the Borg column.

    Returns:
    -------
    None
    """
    # Load the features and repetition files
    features_data = pd.read_csv(features_file)
    repetition_data = pd.read_csv(repetition_file)

    # Merge features file with repetition data to include Borg column
    merged_data = pd.merge(
        features_data,
        repetition_data[["Subject", "Repetition", "Borg"]],
        on=["Subject", "Repetition"],
        how="left"  # Use left join to retain all features
    )

    # Save the updated features file
    merged_data.to_csv(output_file, index=False)
    print(f"Updated features file with Borg column saved to {output_file}.")


def merge_features(feature_files, output_file="OutputCSVFiles/merged_features_with_borg.csv"):
    """
    Merges multiple feature files on 'Subject', 'Repetition', and 'Borg' columns.

    Parameters:
    - feature_files (list): List of file paths to the feature CSV files to be merged.
    - output_file (str): Path to save the merged CSV file (default: 'OutputCSVFiles/merged_features_with_borg.csv').

    Returns:
    - None: Saves the merged dataset to the specified output file.
    """
    if not feature_files or len(feature_files) < 2:
        raise ValueError("At least two feature files are required for merging.")

    # Read the first file
    merged_df = pd.read_csv(feature_files[0])
    print(f"Loaded file: {feature_files[0]}")

    # Iteratively merge with the remaining files
    for file in feature_files[1:]:
        print(f"Merging with file: {file}")
        current_df = pd.read_csv(file)
        merged_df = pd.merge(merged_df, current_df, on=['Subject', 'Repetition'], how='inner')

    # Save the merged dataset
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to: {output_file}")


def map_borg_to_features():
    # Load Borg data
    borg_df = pd.read_csv('Borg data/borg_data.csv')

    # Filter for task1_35i
    borg_task1 = borg_df[borg_df['task_order'] == 'task1_35i'].reset_index(drop=True)

    # Melt time columns (10_sec, 20_sec, etc.) into rows
    borg_melted = pd.melt(
        borg_task1,
        id_vars=['subject', 'task_order', 'before_task'],
        var_name='Time',
        value_name='BORG'
    )

    # Filter out non-numeric time columns (e.g., 'end_of_trial')
    borg_melted = borg_melted[borg_melted['Time'].str.contains(r'^\d+_sec$', na=False)]

    # Clean time values (convert "10_sec" to 10, etc.)
    borg_melted['Time'] = borg_melted['Time'].str.replace('_sec', '').astype(int)

    # Include the "before_task" Borg value at time=0
    borg_time_0 = pd.DataFrame({
        'subject': borg_task1['subject'],
        'task_order': 'task1_35i',
        'Time': 0,
        'BORG': borg_task1['before_task']
    })
    borg_combined = pd.concat([borg_time_0, borg_melted], ignore_index=True)

    # Sort by subject and time
    borg_combined = borg_combined.sort_values(['subject', 'Time'])
    print(borg_combined)

    features_df = pd.read_csv('Features/Extracted/allmerged_features.csv')

    # Define time per repetition (replace with your actual value)
    TIME_PER_REPETITION = 3  # seconds

    # Calculate cumulative time for each repetition
    features_df['Time'] = features_df['Repetition'] * TIME_PER_REPETITION

    def interpolate_borg(group):
        subject = group['Subject'].iloc[0]

        # Get Borg data for this subject
        subject_borg = borg_combined[borg_combined['subject'] == f'subject_{subject}']
        if subject_borg.empty:
            return group  # Skip if no Borg data

        # Extract time and Borg values
        times = subject_borg['Time'].values
        borgs = subject_borg['BORG'].values

        # Interpolate Borg values for the group's time points
        group['BORG'] = np.interp(
            group['Time'],
            times,
            borgs,
            left=np.nan,  # Assign NaN if time < first Borg recording
            right=np.nan  # Assign NaN if time > last Borg recording
        )

        # Exclude samples exceeding the final Borg time by 10 seconds
        max_borg_time = times[-1]
        group = group[group['Time'] <= max_borg_time + 10]

        return group

    # Apply interpolation for each subject
    features_df = features_df.groupby('Subject', group_keys=False).apply(interpolate_borg)

    # Drop rows where Borg could not be interpolated
    features_df = features_df.dropna(subset=['BORG'])

    features_df.to_csv('Features/Extracted/allmerged_features_with_borg.csv', index=False)


