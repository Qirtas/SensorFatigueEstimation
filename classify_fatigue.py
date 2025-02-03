# model_classifier.py

import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def train_and_evaluate_classification_models(csv_path, results_file, features_imp_file):
    """
    Train and evaluate classification models to predict fatigue (Borg > 13) versus non-fatigue (Borg <= 13).

    Parameters:
    - csv_path (str): Path to the CSV features file.
    - results_file (str): File path to save the performance metrics CSV.
    - features_imp_file (str): File path to save the feature importance CSV.
    """
    print("\n=== Loading Data ===\n")
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from '{csv_path}'.\n")
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Drop rows with missing Borg values
    df = df.dropna(subset=['Borg'])

    # Create binary target: fatigue (1) if Borg > 13, else non-fatigue (0)
    y = (df['Borg'] > 13).astype(int)
    # Remove unneeded columns from the feature set
    X = df.drop(['Subject', 'Repetition', 'Borg'], axis=1)
    groups = df['Subject']

    # Define classification models
    models = {
        'XGBoost Classifier': XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        enable_categorical=True  # Replace use_label_encoder with enable_categorical
    ),
        'Support Vector Classifier': SVC(probability=True, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear')
    }

    # Define GroupKFold (ensure that the number of unique groups is at least the number of splits)
    group_kfold = GroupKFold(n_splits=34)

    # Step 1: Train and evaluate using all features
    # print("=== Training and Evaluating with All Features ===\n")
    # all_features_metrics = train_models(models, X, y, groups, group_kfold)

    # Step 2: Identify top features using feature importance (based on a Random Forest Classifier)
    print("\n=== Identifying Top Features ===\n")
    top_features = get_top_features(X, y, top_n=100, features_imp_file=features_imp_file)
    X_top = X[top_features]

    # Step 3: Train and evaluate using top features
    print("=== Training and Evaluating with Top Features ===\n")
    top_features_metrics = train_models(models, X_top, y, groups, group_kfold)

    # Combine results from both evaluation rounds
    # df_all = pd.DataFrame(all_features_metrics).transpose()
    # df_all.index.name = "Model"
    # df_all["Feature Set"] = "All Features"

    df_top = pd.DataFrame(top_features_metrics).transpose()
    df_top.index.name = "Model"
    df_top["Feature Set"] = "Top Features"

    results_df = pd.concat([df_top])
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file)
    print(f"\nPerformance metrics saved to '{results_file}'.")


def train_models(models, X, y, groups, group_kfold):
    performance_metrics = {}

    for model_name, model in models.items():
        print(f"--- Evaluating {model_name} ---")

        # Define model-specific hyperparameter tuning
        if model_name == 'Support Vector Classifier':
            param_grid = {
                'classifier__C': [0.1, 1, 10, 50],
                'classifier__gamma': ['scale'],
                'classifier__kernel': ['rbf', 'poly', 'sigmoid']
            }
        elif model_name == 'Logistic Regression':
            param_grid = {
                'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            }
        elif model_name == 'Random Forest Classifier':
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'XGBoost Classifier':
            param_grid = {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__max_depth': [3, 5, 7, 9],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                'classifier__gamma': [0, 0.1, 0.2]
            }
        else:
            param_grid = None

        pipeline = Pipeline([
            ('variance_filter', VarianceThreshold(threshold=0.01)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95) if model_name == 'Support Vector Classifier' else None),
            ('classifier', model)
        ])

        if param_grid:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=group_kfold,
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1
            )
            search.fit(X, y, groups=groups)
            best_model = search.best_estimator_
            print(f"Best Parameters for {model_name}: {search.best_params_}")
        else:
            best_model = pipeline

        # Generate cross-validated predictions
        y_pred = cross_val_predict(best_model, X, y, cv=group_kfold, groups=groups, n_jobs=-1)

        # Get predicted probabilities for ROC AUC (if available)
        try:
            y_pred_proba = cross_val_predict(best_model, X, y, cv=group_kfold, groups=groups, n_jobs=-1, method='predict_proba')
            y_pred_proba = y_pred_proba[:, 1]
        except Exception:
            y_pred_proba = None

        # Compute classification performance metrics per class
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=None)
        recall = recall_score(y, y_pred, average=None)
        f1 = f1_score(y, y_pred, average=None)

        performance_metrics[model_name] = {
            'Accuracy': accuracy,
            'Precision': list(precision),
            'Recall': list(recall),
            'F1 Score': list(f1)
        }

        # Print metrics for both classes
        print(f"{model_name} Performance Metrics:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        for cls in range(len(precision)):
            print(f"Class {cls} - Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}, F1 Score: {f1[cls]:.4f}")

    return performance_metrics


def get_top_features(X, y, top_n=100, features_imp_file=None):
    """
    Identify the top features using feature importances from a Random Forest Classifier.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Binary target variable.
    - top_n (int): Number of top features to select.
    - features_imp_file (str): File path to save the feature importance CSV.

    Returns:
    - top_features (list): List of selected top feature names.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('variance_filter', VarianceThreshold(threshold=0.01)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X, y)
    valid_feature_mask = pipeline.named_steps['variance_filter'].get_support()
    valid_feature_names = X.columns[valid_feature_mask]
    feature_importances = pipeline.named_steps['classifier'].feature_importances_

    if len(valid_feature_names) != len(feature_importances):
        raise ValueError("Mismatch between filtered features and feature importances.")

    feature_importance_df = pd.DataFrame({
        'Feature': valid_feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("Top Features:")
    print(feature_importance_df.head(top_n))

    # Save the top feature importances if a file path is provided
    if features_imp_file is not None:
        os.makedirs(os.path.dirname(features_imp_file), exist_ok=True)
        feature_importance_df.head(top_n).to_csv(features_imp_file, index=False)
        print(f"Feature importances saved to '{features_imp_file}'.")

    top_features = feature_importance_df['Feature'].head(top_n).tolist()
    return top_features

