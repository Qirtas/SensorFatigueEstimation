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




def create_heatmap(
            feature_df,
            title="",
            output_folder="FeatureImportance/Heatmaps",
            top_n_highlight=3,
            aggregated_threshold=0.8
    ):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filename = "Top15Features.png"
        filepath = os.path.join(output_folder, filename)

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
            # cbar_kws={'label': 'Normalized Importance'},
            annot_kws={"fontsize": 14}
        )

        for condition, top_features in top_features_per_condition.items():
            for feature in top_features:
                print(feature)
                row_idx = feature_df.index.get_loc(feature)
                col_idx = feature_df.columns.get_loc(condition)
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=2))

                for text in ax.texts:
                    if text.get_position() == (col_idx + 0.5, row_idx + 0.5):  # Match the cell position
                        text.set_text(f"{text.get_text()}*")
                        text.set_fontweight("bold")

        for row_idx, is_high in enumerate(high_aggregated_mask):
            if is_high:
                col_idx = feature_df.columns.get_loc('Aggregated_Importance')
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))


        ax.set_ylabel("", fontsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory
        print(f"Heatmap saved to {filepath}")






