import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_data(base_directory, movement, body_part):
    """
    Load feature data from the specified directory for a specific movement and body part.

    Parameters:
        base_directory (str): Base directory containing feature data.
        movement (str): Movement type (e.g., "Internal", "External").
        body_part (str): Body part sensor (e.g., "Shoulder", "Forearm").

    Returns:
        pd.DataFrame: Combined feature data for all subjects.
    """
    directory_path = os.path.join(base_directory, movement, body_part)
    all_data = []

    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            subject_data = pd.read_csv(file_path)
            subject_data['Subject'] = file.split('_')[1]  # Extract subject ID from the filename
            all_data.append(subject_data)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data


def plot_histograms(data, feature_columns, output_directory, body_part, movement):
    """
    Plot histograms for each feature.

    Parameters:
        data (pd.DataFrame): DataFrame containing the features.
        feature_columns (list): List of feature column names.
        output_directory (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_directory, exist_ok=True)

    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, bins=30, color='blue')
        plt.title(f"Histogram of {feature} for {body_part} {movement} Movement")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f"{feature}_histogram.png"))
        plt.close()


def plot_boxplots(data, feature_columns, output_directory, body_part, movement):
    """
    Plot boxplots for each feature.

    Parameters:
        data (pd.DataFrame): DataFrame containing the features.
        feature_columns (list): List of feature column names.
        output_directory (str): Directory to save the plots.

    Returns:
        None
    """
    os.makedirs(output_directory, exist_ok=True)

    for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[feature], color='orange')
        plt.title(f"Boxplot of {feature} for {body_part} {movement} Movement")
        plt.xlabel(feature)
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f"{feature}_boxplot.png"))
        plt.close()


def visualize_features(base_directory, movement, body_part):
    """
    Visualize feature distributions using histograms and boxplots.

    Parameters:
        base_directory (str): Base directory containing feature data.
        movement (str): Movement type (e.g., "Internal", "External").
        body_part (str): Body part sensor (e.g., "Shoulder", "Forearm").

    Returns:
        None
    """
    print(f"Loading data for {movement} movement and {body_part}...")
    data = load_feature_data(base_directory, movement, body_part)

    feature_columns = [col for col in data.columns if col not in ['Subject', 'Repetition']]  # Exclude metadata columns
    output_directory = os.path.join("Visualizations", movement, body_part)

    print("Generating histograms...")
    plot_histograms(data, feature_columns, os.path.join(output_directory, "Histograms"), body_part, movement)

    print("Generating boxplots...")
    plot_boxplots(data, feature_columns, os.path.join(output_directory, "Boxplots"), body_part, movement)

    print(f"Visualizations saved in {output_directory}")
