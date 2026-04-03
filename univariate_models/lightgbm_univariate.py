"""
This script performs direct 1-step (default) LightGBM autoregressive forecasts with an expanding-window re-training approach and hv-block CV. Additionally, SHAP values are computed for a subsequent explainability analysis.
Fur further methodological details and motivation behind methodological decisions, please refer to the thesis.

Output:
- 1_lightgbm_univariate_forecasts.csv
- 1_lightgbm_univariate_shap_values.csv
- 1_lightgbm_univariate_best_params.csv

- printed RMSE and MAE and Hit Rate in log-diff units for different regimes and naive benchmark

Features:
- target variable (INDPRO) uses first `nlags` lags (default=12)
- an INDPRO uncertainty indicator through a rolling (3-month) standard deviation of INDPRO

Usage in CLI:
    python univariate_models/lightgbm_univariate.py --input data_intermediate/transformed_univariate_1.csv --datecol sasdate --start-train-months 100
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
# importing the helper functions from the separate script
from model_helper import load_series, create_direct_lag_features, hv_block_cv, REGIMES


# main forecasting + SHAP routine
def compute_shap_and_forecasts(
    series_logdiff,
    target_col="INDPRO",
    nlags=12,
    horizon=1,
    initial_train_months=100,
    cv_interval=12,
    h=1,
    v=24,
    verbose=True
):
    start_time = time.time()
    n = len(series_logdiff)

    shap_records = []
    forecast_records = []
    y_true_all, y_pred_all, dates_all = [], [], []
    # creating a storage of all best hyperparameter subsets
    best_params_records = []


    # various tested hyperparameter sets shown below - the last one was used for obtaining the final model results 

#     raw_params = [
#     {
#         "n_estimators": [100, 500],           
#         "learning_rate": [0.01, 0.05],       
#         "num_leaves": [7, 15, 31],            
#         "max_depth": [3, 5, -1],              
#         "min_child_samples": [5, 10, 20],     
#         "reg_alpha": [0.1, 0.5],              
#         "reg_lambda": [0.1, 0.5],             
#         "subsample": [0.8],              
#         "colsample_bytree": [0.8],           
#         "importance_type": ["gain"],          
#         "random_state": [42],
#         "n_jobs": [-1],
#         "verbosity": [-1]                    
#     }
# ]
#     raw_params = [
#     {
#         "n_estimators": [50, 100, 300],       
#         "learning_rate": [0.005, 0.01, 0.03],
#         "num_leaves": [5, 7, 12],           
#         "max_depth": [2, 3, 5],               
#         "min_child_samples": [10, 20, 30],    
#         "reg_alpha": [0, 0.5, 2.0],           
#         "reg_lambda": [0.1, 1.0, 5.0],        
#         "subsample": [0.7],                  
#         "colsample_bytree": [0.7, 0.8],           
#         "importance_type": ["gain"],
#         "random_state": [42],
#         "n_jobs": [-1],
#         "verbosity": [-1],
#         "objective":["huber"]
#     }
# ]
    # hyperparameter grid
    raw_params = [
    {
        "n_estimators": [100, 500],           # number of boosting rounds
        "max_depth": [3, 5, -1],              # shallow trees for macro data
        "subsample": [0.8],                   # row bagging
        "colsample_bytree": [0.8],            # feature bagging
        "importance_type": ["gain"],          # better for SHAP consistency
        "random_state": [42],
        "verbosity": [-1]                    
    }
    ]

    param_grid = list(ParameterGrid(raw_params))
    
    best_params = param_grid[0]

    # default parameters - irrelevant as first cycle is performed instantly
    best_params = param_grid[0]

     # for each origin date
    for origin_idx in range(initial_train_months, n - horizon + 1):

        ###################################### rolling window approach inferior
        #window_size = 90 # 10 years
        #train_start = max(0, origin_idx - window_size)
        #current_history = series_logdiff.iloc[train_start:origin_idx] 
        ######################################
        
        # history up to current origin (expanding window)
        current_history = series_logdiff.iloc[:origin_idx]
        df_features = create_direct_lag_features(current_history, nlags=nlags, horizon=horizon)
        
        # training data: rows where target is known
        train_df = df_features.dropna(subset=['target'])
        X_train = train_df.drop(columns=["target", "y"]).values
        y_train = train_df["target"].values
        
        # live feature - last row (target is NaN)
        x_live = df_features.drop(columns=["target", "y"]).iloc[[-1]].values
        origin_date = df_features.index[-1]

        # CV on subset (fixed-length rolling CV window for regime detection) - did not improve performance
        # only using the last 100 months to pick the best parameters
        # cv_lookback = 100 
        # X_cv = X_train[-cv_lookback:]
        # y_cv = y_train[-cv_lookback:]

        
        # HV-block CV for hyperparameter tuning
        if (origin_idx - initial_train_months) % cv_interval == 0:
            tuned_params = hv_block_cv(LGBMRegressor, X_train, y_train, param_grid, h=h, v=v, n_splits=1, winsorize = True)
            if tuned_params is not None:
                best_params = tuned_params
                # record best params for this CV cycle
                record = {
                    "origin_idx": origin_idx,
                    "origin_date": origin_date,
                }
                record.update(best_params)
                best_params_records.append(record)

                if verbose:
                    print(f"CV updated @ {origin_date.date()} → {best_params}")

  
        # winsorization
        # calculating the 1st and 99th percentile of the TRAINING data only
        lower_limit = np.percentile(X_train, 1, axis=0)
        upper_limit = np.percentile(X_train, 99, axis=0)

        # apply the caps to the training data
        X_train = np.clip(X_train, lower_limit, upper_limit)
        
        # apply the same limits to the live feature (prevents extrapolation)
        x_live = np.clip(x_live, lower_limit, upper_limit)
        
        # winsorize the target y_train
        y_low, y_high = np.percentile(y_train, 1), np.percentile(y_train, 99)
        y_train = np.clip(y_train, y_low, y_high)

        model = Pipeline([
            ("scaler", RobustScaler()),
            ("lgbm", LGBMRegressor(**best_params))
        ])


        model.fit(X_train, y_train)
        forecast_val = model.predict(x_live)[0]
        
              
        actual_idx = origin_idx + horizon - 1  # aligned with horizon
        actual_val = series_logdiff.iloc[actual_idx]

        forecast_date = series_logdiff.index[actual_idx]

        y_true_all.append(actual_val)
        y_pred_all.append(forecast_val)
        dates_all.append(forecast_date)

        forecast_records.append({
            "origin_idx": origin_idx,
            "origin_date": origin_date,
            "forecast_date": forecast_date,
            "forecast_logdiff": forecast_val,
            "actual_logdiff": actual_val
        })



        # access the individual steps from the pipeline
        scaler = model.named_steps["scaler"]
        lgbm = model.named_steps["lgbm"]

        X_bg = scaler.transform(X_train[-100:]) # background data for SHAP

        explainer = shap.TreeExplainer(lgbm, X_bg)
        shap_vals = explainer.shap_values(scaler.transform(x_live))


        # getting the feature names from the dataframe to ensure correct mapping
        feature_names = df_features.drop(columns=["target", "y"]).columns.tolist()
        
        # using [0] because x_live and shap_vals are (1, n_features)
        for i, col_name in enumerate(feature_names):
            shap_records.append({
                "origin_date": origin_date,
                "forecast_date": forecast_date,
                "feature": col_name,
                "feature_value": x_live[0, i],
                "shap_value": shap_vals[0, i],
                "prediction": forecast_val,
                "base_value": explainer.expected_value
            })

        if verbose and (origin_idx - initial_train_months) % 12 == 0: # yearly logging update
            print(f"Processed origin {origin_date.date()}")

    
    # metric computation section
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    dates_all = pd.to_datetime(dates_all)

    # naive forecast: constant zero as reference
    y_pred_naive_all = np.roll(y_true_all, 1)

    def hit_rate(y_true, y_pred):
        return np.mean(np.sign(y_true) == np.sign(y_pred))

    # naive persistence forecast
    def custom_naive_mae(y_true, y_naive_pred):
        """
        calculates MAE for the persistence forecast using N-1 denominator to further use it for computing MASE.
        Skips the first observation because it has no previous value
        """
        if len(y_true) <= 1:
            return np.nan
        
        abs_errors = np.abs(y_true[1:] - y_naive_pred[1:])
        return np.sum(abs_errors) / (len(y_true) - 1)

    metrics_dict = {}

    # overall
    metrics_dict["RMSE_full"] = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    metrics_dict["MAE_full"] = mean_absolute_error(y_true_all, y_pred_all)
    metrics_dict["HitRate_full"] = hit_rate(y_true_all, y_pred_all)

    metrics_dict["NAIVE_RMSE_full"] = np.sqrt(mean_squared_error(y_true_all[1:], y_pred_naive_all[1:]))
    metrics_dict["NAIVE_MAE_full"] = custom_naive_mae(y_true_all, y_pred_naive_all)
    metrics_dict["NAIVE_HitRate_full"] = hit_rate(y_true_all[1:], y_pred_naive_all[1:])

    # by regime
    for regime_name, (start_str, end_str) in REGIMES.items():
        mask = (dates_all >= pd.to_datetime(start_str)) & (dates_all <= pd.to_datetime(end_str))

        if mask.any():
            y_true_reg = y_true_all[mask]
            y_pred_reg = y_pred_all[mask]
            y_pred_naive_reg = y_pred_naive_all[mask]

            metrics_dict[f"RMSE_{regime_name}"] = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
            metrics_dict[f"MAE_{regime_name}"] = mean_absolute_error(y_true_reg, y_pred_reg)
            metrics_dict[f"HitRate_{regime_name}"] = hit_rate(y_true_reg, y_pred_reg)

            metrics_dict[f"NAIVE_RMSE_{regime_name}"] = np.sqrt(mean_squared_error(y_true_reg, y_pred_naive_reg))
            metrics_dict[f"NAIVE_MAE_{regime_name}"] = custom_naive_mae(y_true_reg, y_pred_naive_reg)
            metrics_dict[f"NAIVE_HitRate_{regime_name}"] = hit_rate(y_true_reg, y_pred_naive_reg)       
       
        else:
            for prefix in ["", "NAIVE_"]:
                for m in ["RMSE", "MAE", "HitRate"]:
                    metrics_dict[f"{prefix}{m}_{regime_name}"] = np.nan

    metrics_dict["Time elapsed [s]"] = round(time.time() - start_time, 2)

    # final metrics table
    forecast_metric_names = [k for k in metrics_dict.keys() if not k.startswith("NAIVE_")]

    forecast_metric_names = [
        "RMSE_full", "MAE_full", "HitRate_full",
        *[f"{m}_{r}" for r in REGIMES for m in ["RMSE", "MAE", "HitRate"]],
        "Time elapsed [s]"
    ]

    forecast_values = [metrics_dict[m] for m in forecast_metric_names]

    naive_metric_names = [
        f"NAIVE_{m}" if m != "Time elapsed [s]" else ""
        for m in forecast_metric_names
    ]

    naive_values = [
        metrics_dict.get(f"NAIVE_{m}", np.nan) if m != "Time elapsed [s]" else ""
        for m in forecast_metric_names
    ]

    metrics_df = pd.DataFrame(
        [
            forecast_metric_names,
            forecast_values,
            naive_metric_names,
            naive_values,
        ]
    )

    forecast_df = pd.DataFrame(forecast_records)
    forecast_df = pd.concat([forecast_df, metrics_df], axis=1) # concatenate with summary metrics df
    best_params_df = pd.DataFrame(best_params_records)


    return pd.DataFrame(shap_records), forecast_df, best_params_df

# main function
def main(args):
    series_logdiff = load_series(args.input, args.datecol)

    shap_df, forecast_df, best_params_df = compute_shap_and_forecasts(
        series_logdiff,
        target_col="INDPRO",
        nlags=args.nlags,
        horizon=1,
        initial_train_months=args.start_train_months,
        cv_interval=12,
        h=1,
        v=24,
        verbose=args.verbose
    )

    shap_out = args.output or "results/shap/1_lightgbm_univariate_shap_values.csv"
    shap_df.to_csv(shap_out, index=False)
    print(f"\nSaved SHAP values to {shap_out}")

    forecast_out = "results/forecast/1_lightgbm_univariate_forecasts.csv"
    forecast_df.to_csv(forecast_out, index=False)
    print(f"Saved forecasts to {forecast_out}")

    best_params_out = "results/forecast/1_lightgbm_univariate_best_params.csv"
    best_params_df.to_csv(best_params_out, index=False)

    print(f"\nSaved best hyperparameters per CV cycle to {best_params_out}")

    print("\nSHAP preview:")
    print(shap_df.head())

    print("\nForecast preview:")
    print(forecast_df.head())


# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct LightGBM Regressor AR forecasting with SHAP (1-step only)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--datecol", required=True)
    parser.add_argument("--nlags", type=int, default=12)
    parser.add_argument("--start-train-months", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
