"""
This script performs direct 1-step (default) MLP multivariate forecasts with an expanding-window re-training approach and hv-block CV. Additionally, SHAP values are computed for a subsequent explainability analysis.
Fur further methodological details and motivation behind methodological decisions, please refer to the thesis.

The variables have been selected numerically as described in variable_selection.py.

As shown below, threee CSV are saved: forecasts, SHAP values and the best hyperparameter set for each hyperparameter tuning cycle.

Output:
- 1_mlp_multivariate_shap_values.csv
- 1_mlp_multivariate_forecasts.csv
- 1_mlp_multivariate_best_params.csv

- printed RMSE and MAE and Hit Rate in log-diff units for different regimes and naive benchmark

Features:
- target variable (INDPRO) uses first `nlags` lags (default=12 for consistency with the AR models)
- additional predictors use configurable lags (default=1)
- an INDPRO uncertainty indicator through a rolling (3-month) standard deviation of INDPRO

Usage in CLI:
    python multivariate_models/mlp_multivariate.py --input data_intermediate/transformed_multivariate_domain_1.csv --datecol sasdate --nlags 12 --predictor-lags 1 --start-train-months 100 --verbose
"""


import argparse
import os
import time
import numpy as np
import pandas as pd
import shap
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

# importing the helper functions from the separate script
from model_helper import load_series, create_direct_lag_features_multi, hv_block_cv, REGIMES

# central function for computing forecasts and SHAP values
def compute_shap_and_forecasts(
    df,
    target_col="INDPRO",
    nlags_target=12,
    predictor_lags=None,
    horizon=1,
    initial_train_months=100,
    cv_interval=12,
    h=1,
    v=24,
    verbose=True
):
    start_time = time.time()
    n = len(df)

    # initialize result lists
    shap_records = []
    forecast_records = []
    y_true_all, y_pred_all, dates_all = [], [], []
    # creating a storage of all best hyperparameter subsets
    best_params_records = []

    # default predictor_lags if not provided
    if predictor_lags is None:
        predictor_lags = {col: 1 for col in df.columns if col != target_col} # default is 1 lag per predictor excluding the target autoregressive features


    # hidden_layer_sizes_grid = [1,6,13,27,100]
    # activation_grid = ["relu", "tanh"]
    # alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    # param_grid = [
    #     {
    #         "hidden_layer_sizes": hls,
    #         "activation": act,
    #         "alpha": a,
    #         "max_iter": 5000,
    #         "random_state": 24,
    #         "solver": "lbfgs",
    #         "learning_rate": "adaptive"
    #     }
    #     for hls in hidden_layer_sizes_grid
    #     for act in activation_grid
    #     for a in alpha_grid
    # ]
 
    # hidden_layer_sizes_grid = [1,6,13,27,100,(32, 16)]
    # activation_grid = ["relu", "tanh"]
    # alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    # final hyperparameter grid
    hidden_layer_sizes_grid = [(1,),(9,),(18,),(20,),(37,)]
    
    activation_grid = ["relu", "tanh"]
    
    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1]

    param_grid = [
        {
            "hidden_layer_sizes": hls,
            "activation": act,
            "alpha": a,
            "max_iter": 5000,
            "random_state": 24,
            "solver": "lbfgs",
            "learning_rate": "adaptive"
        }
        for hls in hidden_layer_sizes_grid
        for act in activation_grid
        for a in alpha_grid
    ]

    # default parameters - irrelevant as first cycle is performed instantly
    best_params = param_grid[0]

    # for each origin date
    for origin_idx in range(initial_train_months, n - horizon + 1):
        current_history = df.iloc[:origin_idx]

       # assemble feature through helper function
        df_features = create_direct_lag_features_multi(
            df=current_history,
            target_col=target_col,
            nlags_target=nlags_target,
            predictor_lags=predictor_lags,
            horizon=horizon
        )

        # training data: rows where target is known
        train_df = df_features.dropna(subset=['target'])
        X_train = train_df.drop(columns=["target"]).values
        y_train = train_df["target"].values

        # live feature - last row (target is NaN)
        x_live = df_features.drop(columns=["target"]).iloc[[-1]].values
        origin_date = df_features.index[-1]


        # HV-block CV for hyperparameter tuning
        if (origin_idx - initial_train_months) % cv_interval == 0:
            tuned_params = hv_block_cv(MLPRegressor, X_train, y_train, param_grid,n_splits=1, h=h, v=v, winsorize = True)
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
            ("mlp", MLPRegressor(**best_params))
        ])

       
        model.fit(X_train, y_train)

        forecast_val = model.predict(x_live)[0]

        actual_idx = origin_idx + horizon - 1  # aligned with horizon
        actual_val = df[target_col].iloc[actual_idx]
        forecast_date = df.index[actual_idx]

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
        mlp = model.named_steps["mlp"]

        X_bg = scaler.transform(X_train[-100:]) # background data for SHAP

        explainer = shap.KernelExplainer(mlp.predict, X_bg[-100:])
        shap_vals = explainer.shap_values(scaler.transform(x_live), n_samples = "auto")

        # getting the feature names from the dataframe to ensure correct mapping
        feature_names = df_features.drop(columns=["target"]).columns.tolist()

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

    # naive forecast: constant zero
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
    df = load_series(args.input, args.datecol)

    # predictors = all columns except target - same lag order
    predictor_lags_config = {col: args.predictor_lags for col in df.columns if col != "INDPRO"}

    shap_df, forecast_df, best_params_df = compute_shap_and_forecasts(
        df=df,
        target_col="INDPRO",
        nlags_target=args.nlags,
        predictor_lags=predictor_lags_config,
        horizon=1,
        initial_train_months=args.start_train_months,
        cv_interval=12,
        h=1,
        v=24,
        verbose=args.verbose
    )

    shap_out = args.output or "results/shap/1_mlp_multivariate_shap_values.csv"
    shap_df.to_csv(shap_out, index=False)
    print(f"\nSaved SHAP values to {shap_out}")

    forecast_out = "results/forecast/1_mlp_multivariate_forecasts.csv"
    forecast_df.to_csv(forecast_out, index=False)
    print(f"Saved forecasts to {forecast_out}")
          
    best_params_out = "results/forecast/1_mlp_multivariate_best_params.csv"
    best_params_df.to_csv(best_params_out, index=False)

    print(f"\nSaved best hyperparameters per CV cycle to {best_params_out}")

    print("\nSHAP preview:")
    print(shap_df.head())

    print("\nForecast preview:")
    print(forecast_df.head())

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct MLP multivariate forecasting with SHAP (1-step)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--datecol", required=True)
    parser.add_argument("--nlags", type=int, default=12)
    parser.add_argument("--predictor-lags", type=int, default=1,
                        help="Number of lags for additional predictors")
    parser.add_argument("--start-train-months", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
