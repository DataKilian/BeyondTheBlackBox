"""
This script performs direct 1-step (default) Elastic Net autoregressive forecasts with an expanding-window re-training approach and hv-block CV. Additionally, SHAP values are computed for a subsequent explainability analysis.
Fur further methodological details and motivation behind methodological decisions, please refer to the thesis.

Output:
- 1_elasticnet_univariate_forecasts.csv
- 1_elasticnet_univariate_shap_values.csv
- 1_elasticnet_univariate_best_params.csv

- printed RMSE and MAE and Hit Rate in log-diff units for different regimes and naive benchmark

Features:
- target variable (INDPRO) uses first `nlags` lags (default=12)
- an INDPRO uncertainty indicator through a rolling (3-month) standard deviation of INDPRO

Usage in CLI:
    python univariate_models/elasticnet_univariate.py --input data_intermediate/transformed_univariate_1.csv --datecol sasdate --start-train-months 100
"""


import os
import argparse
import time
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
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


    # hyperparameter grid
    alpha_grid = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    l1_ratio_grid = [0,0.1, 0.3, 0.5, 0.7, 0.9,1]

    param_grid = [
        {
            "alpha": a,
            "l1_ratio": l1,
            "fit_intercept": True,
            "max_iter": 50000,
            "random_state": 24,
        }
        for a in alpha_grid
        for l1 in l1_ratio_grid
    ]

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
            tuned_params = hv_block_cv(ElasticNet, X_train, y_train, param_grid, h=h, v=v, n_splits=1, winsorize = True)
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
        
        # winsorization of the target
        y_low, y_high = np.percentile(y_train, 1), np.percentile(y_train, 99)
        y_train = np.clip(y_train, y_low, y_high)
        
        model = Pipeline([
            ("scaler", RobustScaler()),
            ("elastic", ElasticNet(**best_params))
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
        elastic = model.named_steps["elastic"]

        X_bg = scaler.transform(X_train[-100:]) # background data for SHAP

        explainer = shap.LinearExplainer(elastic, X_bg)
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
    y_pred_naive_all = np.zeros_like(y_true_all)

    def hit_rate(y_true, y_pred):
        return np.mean(np.sign(y_true) == np.sign(y_pred))

    metrics_dict = {}

    # overall
    metrics_dict["RMSE_full"] = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    metrics_dict["MAE_full"] = mean_absolute_error(y_true_all, y_pred_all)
    metrics_dict["HitRate_full"] = hit_rate(y_true_all, y_pred_all)
    print("\nOverall Forecast performance:") 
    for k, v in metrics_dict.items(): 
        print(f"{k}: {v:.4f}" if not np.isnan(v) else f"{k}: NaN")

    metrics_dict["NAIVE_RMSE_full"] = np.sqrt(mean_squared_error(y_true_all, y_pred_naive_all))
    metrics_dict["NAIVE_MAE_full"] = mean_absolute_error(y_true_all, y_pred_naive_all)
    metrics_dict["NAIVE_HitRate_full"] = hit_rate(y_true_all, y_pred_naive_all)

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
            metrics_dict[f"NAIVE_MAE_{regime_name}"] = mean_absolute_error(y_true_reg, y_pred_naive_reg)
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

    shap_out = args.output or "results/shap/1_elasticnet_univariate_shap_values.csv"
    shap_df.to_csv(shap_out, index=False)
    print(f"\nSaved SHAP values to {shap_out}")

    forecast_out = "results/forecast/1_elasticnet_univariate_forecasts.csv"
    forecast_df.to_csv(forecast_out, index=False)
    print(f"Saved forecasts to {forecast_out}")

    best_params_out = "results/forecast/1_elasticnet_univariate_best_params.csv"
    best_params_df.to_csv(best_params_out, index=False)

    print(f"\nSaved best hyperparameters per CV cycle to {best_params_out}")

    print("\nSHAP preview:")
    print(shap_df.head())

    print("\nForecast preview:")
    print(forecast_df.head())


# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct Elastic Net AR forecasting with SHAP (1-step only)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--datecol", required=True)
    parser.add_argument("--nlags", type=int, default=12)
    parser.add_argument("--start-train-months", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
