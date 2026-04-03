"""
This is a repository of all helper functions used in the univariate modeling step of the pipeline.
The functions are getting imported into the specific modeling scripts for further usage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import warnings

# definition of the economic sub-regimes
REGIMES = {
    "pre-GFC": ("1900-01-01", "2007-07-31"),  
    "GFC": ("2007-08-01", "2009-06-30"),
    "post-GFC/pre-COVID": ("2009-07-01", "2020-02-29"),
    "COVID": ("2020-03-01", "2023-04-30"),
    "post-COVID": ("2023-05-01", "2100-01-01")
}


# dataset loading
def load_series(path, date_col):
    df = pd.read_csv(path)

    if df.iloc[0][date_col].startswith("Transform"):
        df = df.iloc[1:].copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.sort_index()

    if "INDPRO" not in df.columns:
        raise ValueError("INDPRO column missing")
    
    # filter from 1st March 1968 to and including September 1st, 2025 - for consistency reasons in this study
    df = df.loc['1968-03-01':'2025-09-01']

    s = df["INDPRO"].astype(float)
    s.name = "INDPRO"  
    return s


def create_direct_lag_features(series, nlags=12, horizon=1):
    
    df = pd.DataFrame({"y": series})

    for lag in range(0, nlags):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["volatility_3m"] = df[["lag_0", "lag_1", "lag_2"]].std(axis=1)

    # cyclical month encoding was tested - no improvement registered
    # df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    # df["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    df["target"] = df["y"].shift(-horizon)

    feature_cols = [c for c in df.columns if c not in ["target", "y"]]
    df = df.dropna(subset=feature_cols)

    return df


# cross validation function based on the hv-block approach
def hv_block_cv(
    model_class,
    X,
    y,
    param_grid,
    n_splits=3,
    h=1,
    v=24,
    winsorize=False
):
    """
    hv-block cross-validation for hyperparameter selection using the scikit-learn TimeSeriesSplit function with a gap

    Parameters
    -------
    model_class : class - e.g., LGBMRegressor
    X : array - feature matrix
    y : array - target vector
    param_grid : list of dict - search space of of hyperparameter dictionaries to evaluate
    n_splits : int - number of validation blocks (splits)
    v : int - Validation block size
    h : int - number of samples to exclude between train and validation (h-gap)

    Output
    -------
    best_params : dict - hyperparameters with the lowest average validation MAE
    """
    # ingnore warnings that arise due to non-optimal hyperparameter configurations and keep the terminal clean
    warnings.filterwarnings("ignore")

    best_mae = np.inf # initializing default MAE as +inf
    best_params = None
    n = len(X)

    # assess, if enough observations are available to perform the desired splitting procedure
    min_required = n_splits * v + h + 1
    use_tscv = n >= min_required and n_splits >= 2


    if use_tscv:
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=v,
            gap=h
        )

        splits = list(tscv.split(X))
    else: # fallback to manual single hv-block cross-validation
        val_end = n
        val_start = max(0, val_end - v)
        train_end = max(0, val_start - h)

        if train_end <= 0:
            return None

        splits = [(
            np.arange(train_end),
            np.arange(val_start, val_end)
        )]

    # looping over the hyperparameter search space
    for params in param_grid:
        maes = []

        for train_idx, val_idx in splits:


            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if winsorize == True:
                # calculating limits per feature based only on training fold
                lower_limits = np.percentile(X_train, 1, axis=0)
                upper_limits = np.percentile(X_train, 99, axis=0)

                # clip both training and validation features
                X_train = np.clip(X_train, lower_limits, upper_limits)
                X_val = np.clip(X_val, lower_limits, upper_limits)

                # winsorize the target as well
                y_low, y_high = np.percentile(y_train, 1), np.percentile(y_train, 99)
                y_train = np.clip(y_train, y_low, y_high)

            if winsorize == "huber": # special winsorization approach for model classes based on huber loss optimization --> no target winsorization
                # calculating limits per feature based only on training fold
                lower_limits = np.percentile(X_train, 1, axis=0)
                upper_limits = np.percentile(X_train, 99, axis=0)

                # clip both training and validation features
                X_train = np.clip(X_train, lower_limits, upper_limits)
                X_val = np.clip(X_val, lower_limits, upper_limits)

            # wrap the model in a Pipeline
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model_class(**params))
            ])
            
            pipeline.fit(X_train, y_train)
            preds=pipeline.predict(X_val)

            # evaluate the MAE of the specific hyperparameter set and append it to the list
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)

        avg_mae = np.mean(maes)

        # if minimum MAE - set as the current best hyperparameter set
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    return best_params



