import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_feature_importance_scores(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data = {}
    for file in files:
        condition = file.replace('.csv', '')  # Extract condition name from filename
        df = pd.read_csv(os.path.join(folder_path, file))
        data[condition] = df.set_index('Feature')['Importance']
    return pd.DataFrame(data)


def aggregate_and_select_features(feature_df, top_n=15):
    feature_df['Aggregated_Importance'] = feature_df.sum(axis=1)
    feature_df = feature_df.sort_values(by='Aggregated_Importance', ascending=False)
    top_features = feature_df.nlargest(top_n, 'Aggregated_Importance').index
    return feature_df.loc[top_features]


def normalize_scores(feature_df):
    return feature_df.div(feature_df.max(axis=0), axis=1)




# def create_heatmap(
#             feature_df,
#             title="",
#             output_folder="FeatureImportance/Heatmaps",
#             top_n_highlight=3,
#             aggregated_threshold=0.8
#     ):
#         # Create the output folder if it doesn't exist
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#
#         filename = "Top15Features.png"
#         filepath = os.path.join(output_folder, filename)
#
#         top_features_per_condition = {}
#         for condition in feature_df.columns[:-1]:  # Exclude 'Aggregated_Importance'
#             top_features_per_condition[condition] = feature_df[condition].nlargest(top_n_highlight).index
#
#         high_aggregated_mask = feature_df['Aggregated_Importance'] >= aggregated_threshold
#
#         plt.figure(figsize=(12, 10))
#         ax = sns.heatmap(
#             feature_df,
#             cmap="magma",
#             annot=True,
#             fmt=".2f",
#             linewidths=0.5,
#             # cbar_kws={'label': 'Normalized Importance'},
#             annot_kws={"fontsize": 14}
#         )
#
#         for condition, top_features in top_features_per_condition.items():
#             for feature in top_features:
#                 print(feature)
#                 row_idx = feature_df.index.get_loc(feature)
#                 col_idx = feature_df.columns.get_loc(condition)
#                 ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=2))
#
#                 for text in ax.texts:
#                     if text.get_position() == (col_idx + 0.5, row_idx + 0.5):  # Match the cell position
#                         text.set_text(f"{text.get_text()}*")
#                         text.set_fontweight("bold")
#
#         for row_idx, is_high in enumerate(high_aggregated_mask):
#             if is_high:
#                 col_idx = feature_df.columns.get_loc('Aggregated_Importance')
#                 ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))
#
#
#         ax.set_ylabel("", fontsize=15)
#         ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#
#         plt.tight_layout()
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         plt.close()  # Close the plot to free up memory
#         print(f"Heatmap saved to {filepath}")






def create_heatmap(
        feature_df,
        title="",
        output_folder="FeatureImportance/Heatmaps",
        top_n_highlight=3,
        aggregated_threshold=0.8
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = "Top15Features.png"
    filepath = os.path.join(output_folder, filename)

    # Define a dictionary mapping feature names to their units
    feature_units = {
        "Z_Upperarm_acc_Mean": "m/s²",
        "X_Upperarm_acc_Mean": "m/s²",
        "Y_Forearm_acc_Min": "m/s²",
        "Y_Forearm_acc_Mean": "m/s²",
        "emg_infraspinatus_WV_Slope":"µV/s",
        "Y_Shoulder_acc_Mean": "m/s²",
        "Z_Shoulder_acc_Mean": "m/s²",
        "Z_Palm_acc_Mean": "m/s²",
        "emg_pectoralis_major_STFT_LowFreqPower": "Hz",
        "emg_pectoralis_major_Range": "µV",
        "Z_Palm_acc_Min": "m/s²",
        "emg_deltoideus_posterior_TimeToHalfWV": "s",
        "X_Palm_gyr_Energy_Mid": "rad²/s²",
        "emg_trapezius_ascendens_STFT_MDF": "Hz",
        "X_Torso_acc_Mean": "m/s²",
        "Y_Torso_acc_Mean": "m/s²"

    }

    # Add units to feature names
    feature_df = feature_df.rename(index=lambda f: f"{f} ({feature_units[f]})" if f in feature_units else f)

    # Extract top features per condition for annotation
    top_features_per_condition = {}
    for condition in feature_df.columns[:-1]:  # Exclude 'Aggregated_Importance'
        top_features_per_condition[condition] = feature_df[condition].nlargest(top_n_highlight).index

    high_aggregated_mask = feature_df['Aggregated_Importance'] >= aggregated_threshold

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        feature_df,
        cmap="magma",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"fontsize": 14}
    )

    # Highlight top features
    for condition, top_features in top_features_per_condition.items():
        for feature in top_features:
            row_idx = feature_df.index.get_loc(feature)
            col_idx = feature_df.columns.get_loc(condition)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=2))

            for text in ax.texts:
                if text.get_position() == (col_idx + 0.5, row_idx + 0.5):  # Match the cell position
                    text.set_text(f"{text.get_text()}*")
                    text.set_fontweight("bold")

    # Highlight features with high aggregated importance
    for row_idx, is_high in enumerate(high_aggregated_mask):
        if is_high:
            col_idx = feature_df.columns.get_loc('Aggregated_Importance')
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))

    ax.set_ylabel("", fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {filepath}")
