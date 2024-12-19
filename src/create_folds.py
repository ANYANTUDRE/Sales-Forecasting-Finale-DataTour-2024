# import pandas and model_selection module of scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold
import os, config


def bin_to_mean_map(df, num_bins = 9):
    sorted_df = df[['quantite_vendue']].sort_values(by='quantite_vendue')
    bin_size = len(sorted_df) // num_bins

    bin_labels = np.repeat(range(num_bins), bin_size)
    remaining = len(sorted_df) % num_bins
    if remaining > 0:
        bin_labels = np.append(bin_labels, np.repeat(num_bins - 1, remaining))

    sorted_df['quantite_vendue_binned'] = bin_labels
    df['quantite_vendue_binned'] = sorted_df['quantite_vendue_binned']

    bin_means = df.groupby('quantite_vendue_binned')['quantite_vendue'].mean()
    bin_to_mean_map = dict(enumerate(bin_means))
    df['quantite_vendue_mapped'] = df['quantite_vendue_binned'].map(bin_to_mean_map)
    return df


if __name__ == "__main__":
    df = pd.read_csv(config.ORIG_TRAIN_FILE, index_col=0)
    
    df.dropna(subset=['quantite_vendue'], inplace=True)
    df = df.reset_index(drop=True)
    df['quantite_vendue'] = df['quantite_vendue'].astype(int)
    df = bin_to_mean_map(df, num_bins = 9)
    #print(df.head())
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # fetch targets (if applicable for supervised problems)
    y = df['quantite_vendue_binned'].values

    # initiate the TimeSeriesSplit class
    tscv = KFold(n_splits=config.N_FOLDS, random_state=42)

    # fill the new kfold column
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df, y)):
        df.loc[val_idx, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv(os.path.join(config.DATA_INPUT, "train_folds.csv"), index=False)
    print("CREATED TIME SERIES FOLDS SUCCESSFULLY!!!")
