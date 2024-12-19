import pandas as pd
import numpy as np

def post_process(preds):
     return np.array([int(pred) for pred in preds])

def label_encoding(df):
    df["Gender"] = df["Gender"].map({"Female":0, "Male":1})
    df["Working Professional or Student"] = df["Working Professional or Student"].map({"Working Professional":0, "Student":1})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"No":0, "Yes":1})
    df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"No":0, "Yes":1})
    return df

def date_to_datetime_index(train, submission):
    train['original_date'] = train['date']
    submission['original_date'] = submission['date']

    train      = train.set_index('date')
    submission = submission.set_index('date')

    train.index      = pd.to_datetime(train.index)
    submission.index = pd.to_datetime(submission.index) 
    return train, submission

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


def _fill_missing_id_produit(df):
    frequent_ids = (
        df.dropna(subset=['id_produit'])
        .groupby(['categorie', 'marque'])['id_produit']
        .agg(lambda x: x.value_counts().idxmax())  # Get most frequent 'id_produit'
    )
    
    marque_frequency_by_categorie = (
        df.dropna(subset=['marque'])
        .groupby(['categorie', 'marque'])['marque']
        .count()  # Count occurrences of 'marque'
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    def fill_id(row):
        if pd.isna(row['id_produit']):  # Check if 'id_produit' is missing
            key = (row['categorie'], row['marque'])
            if key in frequent_ids:
                return frequent_ids[key]

            most_frequent_marque = marque_frequency_by_categorie.loc[
                marque_frequency_by_categorie['categorie'] == row['categorie']
            ].iloc[0]['marque']
            return frequent_ids.get((row['categorie'], most_frequent_marque), None)
        return row['id_produit']  # Keep original value if not missing

    df['id_produit'] = df.apply(fill_id, axis=1)
    #print(f"Nombre de valeurs manquantes dans 'id_produit' après traitement: {df['id_produit'].isna().sum()}")
    return df

def _fix_weekend_column(df):
    df['weekend'] = df['weekend'].fillna(
      (df.index.to_series().dt.weekday >= 5).astype(int)
    )
    return df

def _verify_and_impute_jour_ferie(df):
    # Vérification des valeurs non manquantes
    non_missing = df[~df['jour_ferie'].isna()]
    incorrect_rows = non_missing[(non_missing['jour_ferie'] == 1) & (non_missing.index.month != 12)]

    if not incorrect_rows.empty:
        print("Assomption non vérifiée. Voici les lignes où 'jour_ferie' = 1 en dehors de décembre:")
        print(incorrect_rows)
        return df  # Retourne le DataFrame d'origine si l'assomption est fausse
    #print("Assomption vérifiée: 'jour_ferie' = 1 uniquement en décembre pour les valeurs non manquantes.")
    is_december = (df.index.month == 12) # Créer une série booléenne pour les mois de décembre
    missing_indices = df['jour_ferie'].isna()
    df.loc[missing_indices, 'jour_ferie'] = is_december[missing_indices].astype(float) # Imputation des valeurs manquantes
    return df


def fill_missing_values_specific_cols(df):
    ### MISSING VALUES FILLING FOR SPECIFIC COLS
    # id_produit
    df = _fill_missing_id_produit(df)

    # weekend
    df = _fix_weekend_column(df)

    # jour_ferie
    df['jour_ferie'] = df['jour_ferie'].astype(float)  # S'assurer du type float si nécessaire
    df = _verify_and_impute_jour_ferie(df)
    #print(f"Valeurs manquantes dans 'jour_ferie': {df['jour_ferie'].isna().sum()}")
    return df


def fillNaFeaturesRelatedToIdProduit(X_train, X_test):
    train_id_produit = X_train['id_produit']
    test_id_produit = X_test['id_produit']

    # Fill 'id_produit' NaNs with corresponding 'categorie' values
    X_train['id_produit'] = X_train['id_produit'].fillna(X_train['categorie'])
    X_test['id_produit'] = X_test['id_produit'].fillna(X_test['categorie'])

    # Fill 'prix_unitaire' based on 'id_produit' group in training set
    train_grouped_means = X_train.groupby('id_produit')['prix_unitaire'].mean()
    X_train['prix_unitaire'] = X_train.apply(
        lambda row: train_grouped_means[row['id_produit']] if pd.isna(row['prix_unitaire']) else row['prix_unitaire'],
        axis=1
    )
    X_test['prix_unitaire'] = X_test.apply(
        lambda row: train_grouped_means[row['id_produit']] if pd.isna(row['prix_unitaire']) else row['prix_unitaire'],
        axis=1
    )

    # Fill 'marque' based on 'id_produit' group in training set
    train_grouped_modes = X_train.groupby('id_produit')['marque'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
    X_train['marque'] = X_train.apply(
        lambda row: train_grouped_modes[row['id_produit']] if pd.isna(row['marque']) else row['marque'],
        axis=1
    )
    X_test['marque'] = X_test.apply(
        lambda row: train_grouped_modes[row['id_produit']] if pd.isna(row['marque']) else row['marque'],
        axis=1
    )

    # Fill 'stock_disponible' based on 'id_produit' group in training set
    train_grouped_stock_means = X_train.groupby('id_produit')['stock_disponible'].mean()
    X_train['stock_disponible'] = X_train.apply(
        lambda row: train_grouped_stock_means[row['id_produit']] if pd.isna(row['stock_disponible']) else row['stock_disponible'],
        axis=1
    )
    X_test['stock_disponible'] = X_test.apply(
        lambda row: train_grouped_stock_means[row['id_produit']] if pd.isna(row['stock_disponible']) else row['stock_disponible'],
        axis=1
    )
    X_train['id_produit'] = train_id_produit
    X_test['id_produit'] = test_id_produit

    return X_train, X_test

def _create_features(df):
    """
    Create time series features based on time series index and additional date-related features.
    """
    df = df.copy()

    # Existing features
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Additional features from the getDateFeature function
    #df['is_month_start'] = df.index.is_month_start.astype(int)
    #df['is_month_end'] = df.index.is_month_end.astype(int)
    #df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    #df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    #df['is_year_start'] = df.index.is_year_start.astype(int)
    #df['is_year_end'] = df.index.is_year_end.astype(int)
    #df['is_leap_year'] = df.index.is_leap_year.astype(int)
    return df

def _add_season_flags(df):
    # Example for a generic African season classification
    conditions = [
        df['month'].isin([6, 7, 8, 9]),  # Rainy Season
        df['month'].isin([10, 11, 12, 1]),  # Dry Season
        df['month'].isin([2, 3, 4, 5])  # Transition Season (harmattan or cooler)
    ]
    choices = [0, 1, 2]
    df['season'] = np.select(conditions, choices, default='Unknown')
    df['season'] = df['season'].astype(int)
    return df

def _promotion_stock_ratio(df):
    df['promotion']=df['promotion']+1
    df['promotion_stock_ratio'] = np.log1p(df['stock_disponible']) / df['promotion']
    return df

# Function to parse 'id_produit'
def _extract_features_from_id_produit(df):
    # Split id_produit into components
    df[['cat_prefix', 'arrival_date', 'product_number']] = df['id_produit'].str.split('-', expand=True)

    # Parse arrival month and year
    df['arrival_year'] = df['arrival_date'].str[-4:].astype(int)
    df['arrival_month'] = df['arrival_date'].str[:2].astype(int)

    # Create arrival datetime
    df['arrival_datetime'] = pd.to_datetime(
        df['arrival_year'].astype(str) + '-' + df['arrival_month'].astype(str) + '-01'
    )

    # Calculate elapsed days between arrival and sale date
    df['elapsed_days'] = (df.index - df['arrival_datetime']).dt.days

    # Convert product number to numeric
    df['product_number'] = df['product_number'].astype(int)

    # Drop intermediate columns if not needed
    df.drop(['cat_prefix', 'arrival_date', 'arrival_datetime'], axis=1, inplace=True)
    return df


def add_lag_features(df, lag_days, target_col, id_col='id_produit', date_col='date'):
    # Ensure the dataset is sorted by ID and date for lag creation
    df = df.sort_values(by=[id_col, date_col])
    
    for lag in lag_days:
        # Create a new column for each lag
        df[f'{target_col}_lag_{lag}'] = df.groupby(id_col)[target_col].shift(lag)
    
    return df


# Function to add new features to the dataset
def feature_engineering(df):
    # Adding brand new features
    """
    Add engineered features to the dataset for enhanced predictive power.
    """
    # 1. Interaction Features
    df['promotion_weekend'] = df['promotion'] * df['weekend']  # Effect of promotion on weekends
    df['holiday_promotion'] = df['promotion'] * df['jour_ferie']  # Effect of promotion on holidays
    
    # 2. Ratios and Proportions
    df['price_stock_ratio'] = df['prix_unitaire'] / (df['stock_disponible'] + 1)
    
    # 5. Rolling and Lag Features
    #df['rolling_mean_7d'] = df.groupby('id_produit')['quantite_vendue'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    #df['lag_7'] = df.groupby('id_produit')['quantite_vendue'].shift(7)  # Sales 7 days ago

    # Adding lag features
    #lag_days = [1, 2, 3, 4, 5, 6, 7, 30]  # Specify the lags you want
    #df = add_lag_features(df, lag_days=lag_days, target_col='quantite_vendue')
    
    # add features from existing functon
    #df = _create_lag_features(df, 'quantite_vendue', window=3)
    #df = _create_rolling_features(df, 'quantite_vendue', window=3)
    #df = _create_features(df)
    #df = _add_season_flags(df)
    #df = _promotion_stock_ratio(df)
    #df = _extract_features_from_id_produit(df)
    #df = _analyze_and_fill_region_from_quantite_vendue(df, quantile_bins=5)

    return df #, new_cols


def analyze_and_fill_region_from_quantite_vendue(df_train, df_valid, quantile_bins=5):
    """
    Analyze and fill missing 'region' values independently for training and validation data.
    This function ensures no leakage by computing bins and mappings only from training data.
    """
    # Step 1: Train-only binning
    df_train['quantite_bin'], bin_edges = pd.qcut(
        df_train['quantite_vendue'], q=quantile_bins, duplicates='drop', retbins=True
    )

    # Step 2: Compute dominant region for each bin in training data
    dominant_region = (
        df_train.dropna(subset=['region'])
        .groupby('quantite_bin')['region']
        .agg(lambda x: x.value_counts().idxmax())
    )
    region_mapping = dominant_region.to_dict()

    # Step 3: Fill missing regions in training data
    df_train['region'] = df_train.apply(
        lambda row: region_mapping.get(row['quantite_bin'], row['region']) if pd.isna(row['region']) else row['region'],
        axis=1
    )
    df_train.drop('quantite_bin', axis=1, inplace=True)

    # Step 4: Apply binning to validation data using training bin edges
    df_valid['quantite_bin'] = pd.cut(
        df_valid['quantite_vendue'], bins=bin_edges, labels=dominant_region.index, include_lowest=True
    )

    # Step 5: Fill missing regions in validation data
    df_valid['region'] = df_valid.apply(
        lambda row: region_mapping.get(row['quantite_bin'], row['region']) if pd.isna(row['region']) else row['region'],
        axis=1
    )
    df_valid.drop('quantite_bin', axis=1, inplace=True)

    return df_train, df_valid, region_mapping



def analyze_and_fill_promotion_from_quantite_vendue(df_train, df_valid, quantile_bins=10):
    """
    Analyze and fill missing 'promotion' values independently for training and validation data.
    This function ensures no leakage by computing bins and mappings only from training data.
    """
    # Step 1: Train-only binning
    df_train['quantite_bin'], bin_edges = pd.qcut(
        df_train['quantite_vendue'], q=quantile_bins, duplicates='drop', retbins=True
    )

    # Step 2: Compute dominant promotion for each bin in training data
    bin_promotion_counts = (
        df_train.dropna(subset=['promotion'])
        .groupby('quantite_bin')['promotion']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    bin_to_promotion = bin_promotion_counts.idxmax(axis=1).to_dict()

    # Step 3: Fill missing promotions in training data
    df_train['promotion'] = df_train.apply(
        lambda row: bin_to_promotion.get(row['quantite_bin'], row['promotion']) if pd.isna(row['promotion']) else row['promotion'],
        axis=1
    )
    df_train.drop('quantite_bin', axis=1, inplace=True)

    # Step 4: Apply binning to validation data using training bin edges
    df_valid['quantite_bin'] = pd.cut(
        df_valid['quantite_vendue'], bins=bin_edges, labels=bin_promotion_counts.index, include_lowest=True
    )

    # Step 5: Fill missing promotions in validation data
    df_valid['promotion'] = df_valid.apply(
        lambda row: bin_to_promotion.get(row['quantite_bin'], row['promotion']) if pd.isna(row['promotion']) else row['promotion'],
        axis=1
    )
    df_valid.drop('quantite_bin', axis=1, inplace=True)

    return df_train, df_valid, bin_to_promotion
