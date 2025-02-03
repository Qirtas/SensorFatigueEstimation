# model_trainer.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, GroupKFold, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
import datetime
import time
from sklearn.feature_selection import RFE, SelectFromModel
import random
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def train_and_evaluate(csv_path, movementType):
    """
    Train and evaluate models using all features, then use the top 50 features
    from Random Forest to retrain and evaluate models again.

    Parameters:
    - csv_path (str): Path to the feature file.
    - results_file (str): Path to save the performance metrics.
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

    df = df.dropna(subset=['Borg'])

    # Define target and features
    y = df['Borg']
    X = df.drop(['Subject', 'Repetition', 'Borg'], axis=1)
    groups = df['Subject']

    models = {
        'XGBoost Regressor': XGBRegressor(random_state=42),
        'Support Vector Regressor': SVR(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'Elastic Net': ElasticNet(random_state=42)
    }

    group_kfold = GroupKFold(n_splits=32)

    # Step 1: Train and evaluate using all features
    print("=== Training and Evaluating with All Features ===\n")
    # all_features_metrics = train_models(models, X, y, groups, group_kfold, tune_svm=True)

    # Identify top 50 features using Random Forest
    print("\n=== Identifying Top 80 Features ===\n")

    featureImpSavingPath = ''
    if movementType == '35Internal':
        featureImpSavingPath = '35Internal/Results CSVs/features_imp.csv'
        pass
    elif movementType == '45Internal':
        featureImpSavingPath = '45Internal/Results CSVs/features_imp.csv'
        pass
    elif movementType == '55Internal':
        featureImpSavingPath = '55Internal/Results CSVs/features_imp.csv'
        pass
    elif movementType == '35External':
        featureImpSavingPath = '35External/Results CSVs/features_imp.csv'
        pass
    elif movementType == '45External':
        featureImpSavingPath = '45External/Results CSVs/features_imp.csv'
        pass
    elif movementType == '55External':
        featureImpSavingPath = '55External/Results CSVs/features_imp.csv'
        pass
    elif movementType == 'Internal35External35':
        featureImpSavingPath = '35Internal+35External/Results CSVs/features_imp.csv'
        pass
    elif movementType == '35i35e45i45e':
        featureImpSavingPath = '35I+35E+45I+45E/Results CSVs/features_imp.csv'
        pass
    elif movementType == 'Internal45External45':
        featureImpSavingPath = '45Internal+45External/Results CSVs/features_imp.csv'
        pass

    top_features = get_top_features(X, y, featureImpSavingPath, 100)

    # Reduce dataset to top 80 features
    X_top = X[top_features]

    # Step 2: Train and evaluate using top 50 features
    print("=== Training and Evaluating with Top Features ===\n")
    train_models(models, X_top, y, groups, group_kfold, movementType)


def train_models(models, X, y, groups, group_kfold, movementType):
    """
    Train and evaluate models using GroupKFold cross-validation.

    Parameters:
    - models (dict): Dictionary of model names and instances.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - groups (pd.Series): Group labels for GroupKFold.
    - group_kfold (GroupKFold): Cross-validation strategy.
    - tune_svm (bool): Whether to perform hyperparameter tuning for SVM.

    Returns:
    - performance_metrics (dict): Performance metrics for each model.
    """
    performance_metrics = {}

    for model_name, model in models.items():
        print(f"--- Evaluating {model_name} ---")

        if model_name == 'Support Vector Regressor':
            # Perform hyperparameter tuning for SVR
            param_grid = {
                'regressor__C': [0.1, 1, 10, 50],
                'regressor__gamma': ['scale'],
                'regressor__kernel': ['rbf', 'poly', 'sigmoid'],
                'regressor__epsilon': [0.01, 0.1, 0.5]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,  # Reduce the number of iterations for efficiency
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning SVR hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for SVR: {random_search.best_params_}")

        elif model_name == 'Lasso Regression':
            # Perform hyperparameter tuning for Lasso
            param_grid = {
                'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'regressor__max_iter': [1000, 3000],
                'regressor__selection': ['cyclic', 'random'],
                'regressor__tol': [1e-4, 1e-3, 1e-2]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning Lasso hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for Lasso: {random_search.best_params_}")

        elif model_name == 'Ridge Regression':
            # Perform hyperparameter tuning for Ridge
            param_grid = {
                'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                'regressor__tol': [1e-4, 1e-3, 1e-2],
                'regressor__max_iter': [1000, 3000]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning Ridge hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for Ridge: {random_search.best_params_}")

        elif model_name == 'Elastic Net':
            # Perform hyperparameter tuning for Elastic Net
            param_grid = {
                'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'regressor__max_iter': [1000, 3000],
                'regressor__tol': [1e-4, 1e-3, 1e-2],
                'regressor__selection': ['cyclic', 'random']
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning Elastic Net hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for Elastic Net: {random_search.best_params_}")

        elif model_name == 'Random Forest':
            # Perform hyperparameter tuning for Random Forest
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,  # Reduce the number of iterations for efficiency
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning Random Forest hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for Random Forest: {random_search.best_params_}")

        elif model_name == 'XGBoost Regressor':
            # Perform hyperparameter tuning for XGBoost
            param_grid = {
                'regressor__n_estimators': [100, 200, 500],
                'regressor__max_depth': [3, 5, 7, 9],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__min_child_weight': [1, 3, 5],
                'regressor__subsample': [0.8, 0.9, 1.0],
                'regressor__colsample_bytree': [0.8, 0.9, 1.0],
                'regressor__gamma': [0, 0.1, 0.2]
            }

            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=50,  # Reduce the number of iterations for efficiency
                cv=group_kfold,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )

            print("[INFO] Tuning XGBoost hyperparameters...")
            random_search.fit(X, y, groups=groups)

            best_model = random_search.best_estimator_
            print(f"Best Parameters for XGBoost: {random_search.best_params_}")

        else:
            # For models without hyperparameter tuning, use the default model
            pipeline = Pipeline([
                ('variance_filter', VarianceThreshold(threshold=0.01)),
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])

            best_model = pipeline

        # Perform cross-validation predictions
        y_pred = cross_val_predict(
            best_model,
            X,
            y,
            cv=group_kfold,
            groups=groups,
            n_jobs=-1
        )

        # Compute performance metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        mape = (abs((y - y_pred) / y).mean()) * 100

        performance_metrics[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        }

        plotsSavingPath = ''
        if movementType == '35Internal':
            plotsSavingPath = "35Internal/Results CSVs"
            pass
        elif movementType == '45Internal':
            plotsSavingPath = "45Internal/Results CSVs"
            pass
        elif movementType == '55Internal':
            plotsSavingPath = "55Internal/Results CSVs"
            pass
        elif movementType == '35External':
            plotsSavingPath = "35External/Results CSVs"
            pass
        elif movementType == '45External':
            plotsSavingPath = '45External/Results CSVs'
            pass
        elif movementType == '55External':
            plotsSavingPath = '55External/Results CSVs'
            pass
        elif movementType == 'Internal35External35':
            plotsSavingPath = '35Internal+35External/Results CSVs'
            pass
        elif movementType == '35i35e45i45e':
            plotsSavingPath = '35I+35E+45I+45E/Results CSVs'
            pass
        elif movementType == 'Internal45External45':
            plotsSavingPath = '45Internal+45External/Results CSVs'
            pass


        save_actual_vs_predicted_plot(y, y_pred, {model_name}, plotsSavingPath)


        print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%\n")

    return performance_metrics


def save_actual_vs_predicted_plot(y, y_pred, model_name, save_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.5, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line for perfect prediction
    plt.title(f'{model_name}: Actual vs. Predicted BORG Scale')
    plt.xlabel('Actual BORG Scale')
    plt.ylabel('Predicted BORG Scale')

    # Create filename with model name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_actual_vs_predicted_{timestamp}.png"
    full_path = os.path.join(save_path, filename)

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Created/Modified files during execution:
    print(f"Created file: {full_path}")

    return full_path


def get_top_features(X, y, featureImpSavingPath, top_n):

    rf = RandomForestRegressor(random_state=42)
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('variance_filter', VarianceThreshold(threshold=0.01)),
        ('regressor', rf)
    ])

    rf_pipeline.fit(X, y)
    variance_filter = rf_pipeline.named_steps['variance_filter']
    valid_feature_mask = variance_filter.get_support()
    valid_feature_names = X.columns[valid_feature_mask]  # Get only the selected feature names
    rf_model = rf_pipeline.named_steps['regressor']
    feature_importances = rf_model.feature_importances_

    if len(valid_feature_names) != len(feature_importances):
        raise ValueError(f"Feature importances length ({len(feature_importances)}) does not match valid feature names length ({len(valid_feature_names)}).")

    feature_importance_df = pd.DataFrame({
        'Feature': valid_feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("Top Features:")
    print(feature_importance_df.head(top_n))

    top_features = feature_importance_df['Feature'].head(top_n).tolist()
    feature_importance_df.head(top_n).to_csv(featureImpSavingPath, index=False)

    return top_features

