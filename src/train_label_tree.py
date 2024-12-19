import joblib
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing

import config
import model_dispatcher
from preprocessing_utils import (date_to_datetime_index, 
                                fill_missing_values_specific_cols, 
                                fillNaFeaturesRelatedToIdProduit, 
                                feature_engineering, 
                                post_process, 
                                analyze_and_fill_region_from_quantite_vendue, 
                                analyze_and_fill_promotion_from_quantite_vendue
                            )
sys.path.append(os.path.abspath("../"))


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    test = pd.read_csv(config.TEST_FILE, index_col=0)
    df, test = date_to_datetime_index(df, test)

    ### PREPROCESSING MISSING
    df, test = fillNaFeaturesRelatedToIdProduit(df, test)
    df = fill_missing_values_specific_cols(df)

    ############################## HIGH RISK OF LEAKAGE HERE #############################################
    #df, test, _ = analyze_and_fill_region_from_quantite_vendue(df, test, quantile_bins=5)
    #df, test, _ = analyze_and_fill_promotion_from_quantite_vendue(df, test, quantile_bins=5)
    #print(f"Valeurs manquantes restantes: {df.isna().sum()}")
    
    ### FEATURE ENG: adding features from _create_features does't change the score 
    df    = feature_engineering(df)
    test  = feature_engineering(test)

    # Features and target columns
    features = [f for f in df.columns if f not in ("quantite_vendue_binned", "quantite_vendue_mapped", 
                                                   "quantite_vendue", config.TARGET_COL, "kfold",
                                                   "categorie"
                                                   )]
    if fold == 0:
        print(features)

    # Fill NaN values for numerical and categorical columns
    for col in config.NUM_COLS:
        df[col] = df[col].fillna(df[col].mode())
        test[col] = test[col].fillna(df[col].mode())

    for col in features:
        if col not in config.NUM_COLS:
            df[col] = df[col].astype(str).fillna("NONE")
            test[col] = test[col].astype(str).fillna("NONE")

    # Label encoding categorical features
    full_data = pd.concat([df[features], test[features]], axis=0)
    for col in features:
        if col not in config.NUM_COLS:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(full_data[col])
            df[col] = lbl.transform(df[col])
            test[col] = lbl.transform(test[col])

    # Ensure train-validation split respects CV
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    

    ############################## REDUCE RISK OF LEAKAGE#############################################
    # Avoid data leakage: Perform binning and filling separately for train and validation
    df_train, df_valid, _ = analyze_and_fill_region_from_quantite_vendue(df_train, df_valid, quantile_bins=5)
    df_train, df_valid, _ = analyze_and_fill_promotion_from_quantite_vendue(df_train, df_valid, quantile_bins=5)
    ############################################################################


    x_train = df_train[features].values
    x_valid = df_valid[features].values
    test_encoded = test[features].values

    y_train = df_train[config.TARGET_COL].values
    y_valid = df_valid[config.TARGET_COL].values

    # Initialize and fit the model
    # Count target frequencies
    #weights = 1 / df_train[config.TARGET_COL].map(df_train[config.TARGET_COL].value_counts())
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train, 
            #sample_weight=weights
            )

    # Predictions and metrics
    preds = clf.predict(x_valid)
    mape = metrics.mean_absolute_percentage_error(y_valid, preds)
    mape_post = metrics.mean_absolute_percentage_error(y_valid, post_process(preds))
    r2 = metrics.r2_score(y_valid, preds)
    print(f"Fold={fold}, MAPE={mape}, MAPE post={mape_post}, r2={r2}\n")

    # Save model and test set encoding only once
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_{fold}.pkl"))
    if fold == 0:  # Save test encoding only once
        joblib.dump(test_encoded, os.path.join(config.MODEL_OUTPUT, f"test_label_encoded.pkl"), compress=3)

    return mape, mape_post, r2


def run_all_folds(model):
    mapes = []
    mapes_post = []
    r2_scores = []

    print(f"Model: {model}")
    for fold in range(config.N_FOLDS):
        mape, mape_post, r2 = run(fold=fold, model=model)
        mapes.append(mape)
        mapes_post.append(mape_post)
        r2_scores.append(r2)

    avg_mapes = np.mean(mapes)
    avg_mapes_post = np.mean(mapes_post)
    avg_r2 = np.mean(r2_scores)

    print(f"GLOBAL: MAPE={avg_mapes}, MAPE post={avg_mapes_post}, R2={avg_r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    run_all_folds(model=args.model)