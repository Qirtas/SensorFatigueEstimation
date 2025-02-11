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
    print("********")
    print(X.shape[1])
    print(len(X.columns))

    models = {
        'XGBoost Regressor': XGBRegressor(random_state=42),
        'Support Vector Regressor': SVR()
        # 'Random Forest': RandomForestRegressor(random_state=42),
        # 'Ridge Regression': Ridge(random_state=42),
        # 'Lasso Regression': Lasso(random_state=42),
        # 'Elastic Net': ElasticNet(random_state=42)
    }

    group_kfold = GroupKFold(n_splits=32)

    # Step 1: Train and evaluate using all features
    print("=== Training and Evaluating with All Features ===\n")

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
    elif movementType == 'AllMovements':
        featureImpSavingPath = 'AllMovements/Results CSVs/features_imp.csv'
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
            # param_grid = {
            #     'regressor__n_estimators': [100, 200, 500],
            #     'regressor__max_depth': [3, 5, 7, 9],
            #     'regressor__learning_rate': [0.01, 0.05, 0.1],
            #     'regressor__min_child_weight': [1, 3, 5],
            #     'regressor__subsample': [0.8, 0.9, 1.0],
            #     'regressor__colsample_bytree': [0.8, 0.9, 1.0],
            #     'regressor__gamma': [0, 0.1, 0.2]
            # }

            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [5, 7],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__min_child_weight': [5],
                'regressor__subsample': [0.9]}

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
        elif movementType == 'AllMovements':
            plotsSavingPath = 'AllMovements/Results CSVs'
            pass
        elif movementType == 'Internal':
            plotsSavingPath = 'AllMovements/Results CSVs'
            pass
        elif movementType == 'External':
            plotsSavingPath = 'AllMovements/Results CSVs'
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





def train_baseline_model(csv_path, movementType):
    """
    Train a baseline model that predicts the mean Borg value of the training set
    and evaluate its performance for each condition.

    Parameters:
    - csv_path (str): Path to the feature file
    - movementType (str): Type of movement condition being evaluated

    Returns:
    - dict: Dictionary containing the performance metrics
    """
    print("\n=== Training and Evaluating Baseline Model ===\n")

    # Load and prepare data
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from '{csv_path}'.\n")
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = df.bfill().ffill()
    y = df['Borg']
    groups = df['Subject']

    # Create GroupKFold for cross-validation
    group_kfold = GroupKFold(n_splits=32)

    # Initialize lists to store predictions
    y_pred_all = []
    y_true_all = []

    # Perform cross-validation
    for train_idx, test_idx in group_kfold.split(df, y, groups):
        # Get training and test sets
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Calculate mean of training set
        baseline_prediction = np.mean(y_train)

        # Make predictions (same value for all test samples)
        y_pred = np.full(len(y_test), baseline_prediction)

        # Store predictions and true values
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)

    # Convert lists to arrays
    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    # Calculate performance metrics
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_all, y_pred_all)
    mape = np.mean(np.abs((y_true_all - y_pred_all) / y_true_all)) * 100

    # Print results
    print("\n=== Baseline Model Performance ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Create and save actual vs predicted plot
    plotsSavingPath = ''
    if movementType == '35Internal':
        plotsSavingPath = "35Internal/Results CSVs"
    elif movementType == '45Internal':
        plotsSavingPath = "45Internal/Results CSVs"
    elif movementType == '55Internal':
        plotsSavingPath = "55Internal/Results CSVs"
    elif movementType == '35External':
        plotsSavingPath = "35External/Results CSVs"
    elif movementType == '45External':
        plotsSavingPath = '45External/Results CSVs'
    elif movementType == '55External':
        plotsSavingPath = '55External/Results CSVs'
    elif movementType == 'Internal35External35':
        plotsSavingPath = '35Internal+35External/Results CSVs'
    elif movementType == '35i35e45i45e':
        plotsSavingPath = '35I+35E+45I+45E/Results CSVs'
    elif movementType == 'Internal45External45':
        plotsSavingPath = '45Internal+45External/Results CSVs'
    elif movementType == 'AllMovements':
        plotsSavingPath = 'AllMovements/Results CSVs'

    save_actual_vs_predicted_plot(
        pd.Series(y_true_all),
        pd.Series(y_pred_all),
        {'Baseline'},
        plotsSavingPath
    )

    # Return metrics dictionary
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def analyze_borg_errors(y_true, y_pred, condition_name, model_name):
    """
    Analyze prediction errors for each Borg scale value.
    """
    # Create DataFrame with actual and predicted values
    error_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Absolute_Error': np.abs(y_true - y_pred),
        'Percentage_Error': np.abs((y_true - y_pred) / y_true) * 100
    })

    # Group by actual Borg values and calculate statistics
    borg_analysis = error_df.groupby('Actual').agg({
        'Absolute_Error': ['count', 'mean', 'std', 'min', 'max'],
        'Percentage_Error': ['mean', 'std']
    }).round(2)

    # Rename columns for clarity
    borg_analysis.columns = ['Count', 'MAE', 'MAE_STD', 'Min_AE', 'Max_AE',
                           'MAPE', 'MAPE_STD']

    # Sort by MAE to identify problematic Borg values
    borg_analysis_sorted = borg_analysis.sort_values('MAE', ascending=False)

    # Save detailed analysis to CSV
    output_path = f'AllMovements/Results CSVs/borg_error_analysis_{condition_name}_{model_name}.csv'
    borg_analysis_sorted.to_csv(output_path)

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Plot 1: MAE by Borg Value
    plt.subplot(1, 2, 1)
    plt.bar(borg_analysis_sorted.index, borg_analysis_sorted['MAE'])
    plt.title(f'MAE by Borg Value\n{condition_name} - {model_name}')
    plt.xlabel('Actual Borg Value')
    plt.ylabel('Mean Absolute Error')

    # Plot 2: Sample Distribution
    plt.subplot(1, 2, 2)
    plt.bar(borg_analysis_sorted.index, borg_analysis_sorted['Count'])
    plt.title('Sample Distribution')
    plt.xlabel('Borg Value')
    plt.ylabel('Number of Samples')

    plt.tight_layout()
    plt.savefig(f'AllMovements/Results CSVs/borg_error_analysis_{condition_name}_{model_name}.png')
    plt.close()

    return borg_analysis_sorted


def train_all_test_individual(csv_path):
    """
    Train models on all conditions except one using top features and hyperparameter tuning,
    then evaluate on the left-out condition across all subjects at once.
    """
    print("\n=== Loading Combined Data ===\n")
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
    # conditions = df['condition'].unique()
    conditions = ['45External']

    # Initialize models
    models = {
        'XGBoost Regressor': XGBRegressor(random_state=42),
        'Support Vector Regressor': SVR(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    results = {}

    # Leave-One-Condition-Out Cross-Validation
    for condition in conditions:
        print(f"\n=== Testing on Left-Out Condition: {condition} ===")

        train_df = df[df['condition'] != condition]
        test_df = df[df['condition'] == condition]

        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")


        y_train = train_df['Borg']
        X_train = train_df.drop(['Subject', 'Repetition', 'Borg', 'condition'], axis=1)
        y_test = test_df['Borg']
        X_test = test_df.drop(['Subject', 'Repetition', 'Borg', 'condition'], axis=1)

        featureImpSavingPath = f'AllMovements/Results CSVs/features_imp_{condition}.csv'
        top_features = get_top_features(X_train, y_train, featureImpSavingPath, 100)
        X_train = X_train[top_features]
        X_test = X_test[top_features]

        condition_results = {}

        for model_name, model in models.items():
            print(f"\n=== Processing {model_name} ===")

            print("Tuning hyperparameters...")
            tuned_model = tune_model(
                model_name=model_name,
                X=X_train,
                y=y_train,
                groups=train_df['Subject']
            )

            print("Training final model...")
            tuned_model.fit(X_train, y_train)

            print("Testing on left-out condition...")
            y_pred = tuned_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            mape = (abs((y_test - y_pred) / y_test).mean()) * 100

            condition_results[model_name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'predictions': y_pred,
                'actual': y_test.values
            }

            # Perform detailed Borg error analysis
            # borg_analysis = analyze_borg_errors(y_test, y_pred, condition, model_name)

            print(f"\n{model_name} on {condition}:")
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}, MAPE: {mape:.2f}%")

            # print("\nBorg-wise Error Analysis:")
            # print(borg_analysis)

            plot_path = f"AllMovements/Results CSVs/{model_name}_{condition}_test.png"
            save_actual_vs_predicted_plot(y_test.values, y_pred, {model_name}, plot_path)

        results[condition] = condition_results

    return results


def tune_model(model_name, X, y, groups):
    """Generic model tuning function"""
    base_model = {
        'XGBoost Regressor': XGBRegressor(random_state=42),
        'Support Vector Regressor': SVR(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }[model_name]

    param_grid = {
        'XGBoost Regressor': {
            'regressor__n_estimators': [100, 200, 500],
            'regressor__max_depth': [3, 5, 7, 9],
            'regressor__learning_rate': [0.01, 0.05, 0.1]
        },
        'Support Vector Regressor': {
            'regressor__C': [0.1, 1, 10, 50],
            'regressor__gamma': ['scale'],
            'regressor__kernel': ['rbf', 'poly']
        },
        'Random Forest': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

    }[model_name]

    pipeline = Pipeline([
        ('variance_filter', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('regressor', base_model)
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=GroupKFold(n_splits=5),
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X, y, groups=groups)
    return random_search.best_estimator_



def prepare_and_train_models(file_path, top_n=100):  # Added top_n parameter
    # Read the combined dataset
    print("Reading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")

    # Create dictionaries for models with simplified parameters
    models = {
        'XGBoost Regressor': XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'SVR': SVR(kernel='rbf', C=1.0)
        # 'Ridge': Ridge(alpha=1.0),
        # 'Lasso': Lasso(alpha=1.0),
        # 'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }

    # Define number of splits for GroupKFold
    n_splits = 5
    group_kfold = GroupKFold(n_splits=n_splits)

    # Separate internal and external rotations
    internal_conditions = ['35Internal', '45Internal', '55Internal']
    external_conditions = ['35External', '45External', '55External']

    # Train for Internal Rotations
    print("\n=== Training Models for Internal Rotations ===")
    internal_df = df[df['condition'].isin(internal_conditions)].copy()
    print(f"Internal rotations shape: {internal_df.shape}")

    # Clean the data
    internal_df = internal_df.dropna(subset=['Borg'])
    print(f"Internal rotations shape after cleaning: {internal_df.shape}")

    # Prepare features and target for internal rotations
    feature_cols = [col for col in internal_df.columns if col not in ['condition', 'Borg', 'Subject', 'Repetition']]
    X_internal = internal_df[feature_cols]
    y_internal = internal_df['Borg']
    groups_internal = internal_df['Subject']

    # Get top features for internal rotations
    print("\nSelecting top features for internal rotations...")
    internal_top_features = get_top_features(
        X_internal,
        y_internal,
        'AllMovements/Results CSVs/internal_top_features.csv',
        top_n
    )

    # Use only top features for training
    X_internal = X_internal[internal_top_features]

    # Print data information
    print(f"\nFeatures shape after selection: {X_internal.shape}")
    print(f"Target shape: {y_internal.shape}")
    print(f"Number of unique groups: {len(groups_internal.unique())}")

    # Basic data validation after cleaning
    print("\nData validation after cleaning:")
    print(f"X contains inf: {np.any(np.isinf(X_internal))}")
    print(f"X contains nan: {np.any(np.isnan(X_internal))}")
    print(f"y contains inf: {np.any(np.isinf(y_internal))}")
    print(f"y contains nan: {np.any(np.isnan(y_internal))}")

    # Train models for internal rotations
    try:
        internal_metrics = train_models(
            models=models,
            X=X_internal,
            y=y_internal,
            groups=groups_internal,
            group_kfold=group_kfold,
            movementType='Internal'
        )
    except Exception as e:
        print(f"Error in internal rotations training: {str(e)}")
        internal_metrics = None

    # Train for External Rotations
    print("\n=== Training Models for External Rotations ===")
    external_df = df[df['condition'].isin(external_conditions)].copy()

    # Clean external data
    external_df = external_df.dropna(subset=['Borg'])
    print(f"External rotations shape after cleaning: {external_df.shape}")

    # Prepare features and target for external rotations
    X_external = external_df[feature_cols]
    y_external = external_df['Borg']
    groups_external = external_df['Subject']

    # Get top features for external rotations
    print("\nSelecting top features for external rotations...")
    external_top_features = get_top_features(
        X_external,
        y_external,
        'AllMovements/Results CSVs/external_top_features.csv',
        top_n
    )

    # Use only top features for training
    X_external = X_external[external_top_features]

    # Train models for external rotations
    try:
        external_metrics = train_models(
            models=models,
            X=X_external,
            y=y_external,
            groups=groups_external,
            group_kfold=group_kfold,
            movementType='External'
        )
    except Exception as e:
        print(f"Error in external rotations training: {str(e)}")
        external_metrics = None

    # Save results if available
    if internal_metrics:
        results_internal = pd.DataFrame(internal_metrics).T
        results_internal.to_csv('AllMovements/Results CSVs/internal_rotation_results.csv')
        print("\nInternal Rotations Results:")
        print(results_internal)

    if external_metrics:
        results_external = pd.DataFrame(external_metrics).T
        results_external.to_csv('AllMovements/Results CSVs/external_rotation_results.csv')
        print("\nExternal Rotations Results:")
        print(results_external)

    return internal_metrics, external_metrics