import optuna
import joblib
import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from preprocessing_utils import (date_to_datetime_index, 
                                fill_missing_values_specific_cols, 
                                fillNaFeaturesRelatedToIdProduit, 
                                feature_engineering, 
                                post_process, 
                                analyze_and_fill_region_from_quantite_vendue, 
                                analyze_and_fill_promotion_from_quantite_vendue
                            )
from model_dispatcher import models
import config
# ignore warnings ;)
import warnings
warnings.simplefilter("ignore")


def objective(trial, model_name, x_train, x_valid, y_train, y_valid):
    """
    Objective function for Optuna to optimize hyperparameters for a given model.
    """
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'objective': trial.suggest_categorical(
            'objective',
            [
                'reg:squarederror',
                'reg:absoluteerror',
                'reg:squaredlogerror',
                'reg:tweedie'
            ]
        ),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear']),
        'tree_method': trial.suggest_categorical('tree_method', ['hist', 'exact', 'approx', 'auto']),
        'random_state': 42
    }

    # Conditional parameters for Dart Booster
    if params['booster'] == 'dart':
        params.update({
            'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
            'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
            'rate_drop': trial.suggest_uniform('rate_drop', 0.0, 1.0),
            'skip_drop': trial.suggest_uniform('skip_drop', 0.0, 1.0)
        })

    # Conditional parameters for Linear Booster
    if params['booster'] == 'gblinear':
        params.update({
            'lambda': trial.suggest_loguniform('lambda', 1e-4, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-4, 10.0),
            'updater': trial.suggest_categorical('updater', ['shotgun', 'coord_descent']),
            'feature_selector': trial.suggest_categorical('feature_selector', ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty'])
        })

    """
    elif model_name == "hist":
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
        }

    elif model_name == "lgbm":
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        }
    

    if model_name == "bagging":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'max_samples': trial.suggest_uniform('max_samples', 0.1, 1.0),
            'max_features': trial.suggest_uniform('max_features', 0.1, 1.0),
        }

    elif model_name == "stacking":
        params = {
            'final_estimator__max_iter': trial.suggest_int('final_estimator__max_iter', 100, 2000),
            'final_estimator__max_depth': trial.suggest_int('final_estimator__max_depth', 3, 20),
            'final_estimator__learning_rate': trial.suggest_loguniform('final_estimator__learning_rate', 1e-4, 1.0),
        }

    elif model_name == "voting_reg":
        params = {
            'weights': trial.suggest_categorical('weights', [[1, 1], [1, 2], [2, 1]]),
        }"""

    # Initialize model with parameters
    model = models[model_name]
    if params:
        model.set_params(**params)

    # Train and validate the model
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)

    return mape


def run_optuna(model_name, x_train, x_valid, y_train, y_valid, n_trials=500):
    """
    Run Optuna hyperparameter optimization for a given model.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_name, x_train, x_valid, y_train, y_valid), n_trials=n_trials)

    print(f"Best trial for {model_name}: {study.best_trial.params}")
    print(f"Best MAPE for {model_name}: {study.best_value}")

    return study.best_params, study.best_value


def main():
    # Load data
    df = pd.read_csv(config.TRAINING_FILE)
    test = pd.read_csv(config.TEST_FILE, index_col=0)
    df, test = date_to_datetime_index(df, test)

    # Preprocess data
    df, test = fillNaFeaturesRelatedToIdProduit(df, test)
    df = fill_missing_values_specific_cols(df)
    df = feature_engineering(df)
    test = feature_engineering(test)

    # Features and target
    features = [f for f in df.columns if f not in ("quantite_vendue_binned", "quantite_vendue_mapped", 
                                                   "quantite_vendue", config.TARGET_COL, "kfold", "categorie")]
    target = config.TARGET_COL

    # Label encoding categorical features
    full_data = pd.concat([df[features], test[features]], axis=0)
    for col in features:
        if col not in config.NUM_COLS:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(full_data[col])
            df[col] = lbl.transform(df[col])
            test[col] = lbl.transform(test[col])

    # Split data into training and validation
    x_train, x_valid, y_train, y_valid = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    ############################## REDUCE RISK OF LEAKAGE#############################################
    # Avoid data leakage: Perform binning and filling separately for train and validation
    #x_train, x_valid, _ = analyze_and_fill_region_from_quantite_vendue(x_train, x_valid, quantile_bins=5)
    #x_train, x_valid, _ = analyze_and_fill_promotion_from_quantite_vendue(x_train, x_valid, quantile_bins=5)
    ############################################################################

    # Optimize hyperparameters for each model
    results = {}
    for model_name in models.keys():
        print(f"Optimizing {model_name}...")
        best_params, best_mape = run_optuna(model_name, x_train, x_valid, y_train, y_valid)
        results[model_name] = {
            "best_params": best_params,
            "best_mape": best_mape
        }

    # Save results
    results_path = os.path.join(config.MODEL_OUTPUT, "optuna_results.pkl")
    joblib.dump(results, results_path)
    print(f"Optimization results saved to {results_path}")


if __name__ == "__main__":
    main()
