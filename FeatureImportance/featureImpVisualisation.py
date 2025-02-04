import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def load_feature_importance_scores(folder_path):
    """
    Load feature importance scores from multiple CSV files in a folder.
    Each file represents a movement-resistance combination.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data = {}
    for file in files:
        condition = file.replace('.csv', '')  # Extract condition name from filename
        df = pd.read_csv(os.path.join(folder_path, file))
        data[condition] = df.set_index('Feature')['Importance']
    return pd.DataFrame(data)


# Step 2: Aggregate importance scores (mean or max) and select top features
def aggregate_and_select_features(feature_df, top_n=15, aggregation='mean'):
    """
    Aggregate importance scores across conditions and select top features.
    """
    if aggregation == 'mean':
        feature_df['Aggregated_Importance'] = feature_df.mean(axis=1)
    elif aggregation == 'max':
        feature_df['Aggregated_Importance'] = feature_df.max(axis=1)
    else:
        raise ValueError("Aggregation must be 'mean' or 'max'.")

    # Select top N features based on aggregated importance
    top_features = feature_df.nlargest(top_n, 'Aggregated_Importance').index
    return feature_df.loc[top_features]


def normalize_scores(feature_df):
    """
    Normalize importance scores within each condition.
    """
    return feature_df.div(feature_df.max(axis=0), axis=1)




def create_heatmap(
            feature_df,
            title="",
            output_folder="FeatureImportance/Heatmaps",
            top_n_highlight=3,
            aggregated_threshold=0.8
    ):
        """
        Create a heatmap of normalized feature importance scores with highlights for:
        - Top N features for each condition.
        - Features with high aggregated importance scores.

        Parameters:
            feature_df (pd.DataFrame): DataFrame containing normalized feature importance scores.
            title (str): Title of the heatmap.
            output_folder (str): Folder path to save the heatmap.
            top_n_highlight (int): Number of top features to highlight per condition.
            aggregated_threshold (float): Threshold for highlighting high aggregated importance scores.
        """
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate a filename based on the title
        filename = "Top15Features.png"
        filepath = os.path.join(output_folder, filename)

        # Identify top N features for each condition
        top_features_per_condition = {}
        for condition in feature_df.columns[:-1]:  # Exclude 'Aggregated_Importance'
            top_features_per_condition[condition] = feature_df[condition].nlargest(top_n_highlight).index

        # Create a mask for high aggregated importance scores
        high_aggregated_mask = feature_df['Aggregated_Importance'] >= aggregated_threshold

        # Create the heatmap
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            feature_df,
            cmap="magma",  # Diverging color palette, magma
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            # cbar_kws={'label': 'Normalized Importance'},
            annot_kws={"fontsize": 12}  # Adjust annotation font size
        )

        # Highlight top N features for each condition
        for condition, top_features in top_features_per_condition.items():
            for feature in top_features:
                row_idx = feature_df.index.get_loc(feature)
                col_idx = feature_df.columns.get_loc(condition)
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=2))

                # Add asterisks to the annotations for top features
                for text in ax.texts:
                    if text.get_position() == (col_idx + 0.5, row_idx + 0.5):  # Match the cell position
                        text.set_text(f"{text.get_text()}*")
                        text.set_fontweight("bold")

        # Highlight features with high aggregated importance
        for row_idx, is_high in enumerate(high_aggregated_mask):
            if is_high:
                col_idx = feature_df.columns.get_loc('Aggregated_Importance')
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))

        # Adjust font sizes for better readability
        # ax.set_title(title, fontsize=16)
        # ax.set_xlabel("Movement-Resistance Conditions", fontsize=14)
        ax.set_ylabel("", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)

        # Save the plot
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory
        print(f"Heatmap saved to {filepath}")






