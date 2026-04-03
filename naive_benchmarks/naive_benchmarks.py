"""
This script computes the naive benchmark predictive accuracy on a sub-regime, to gain better insights into the suitability of the employed models.

To have a consistent time frame with the models, the input file transformed_multivariate_domain_1.csv is used. The same date restrictions (1st March 1968 to and including September 1st, 2025), as for the models, are inherently applied. 
The inherent application is a result of the fact that the domain-informed multivariate transformed dataset has the smallest number of rows (the most NAs have been dropped from the top of the dataset).
In the modeling scripts, the rows got unified among model types.

Furthermore, the same evaluation time frame (through removal of the first 100 rows, as it is the minimum training sample size for the models) is set.

Input:
- ./data_intermediate/transformed_multivariate_domain_1.csv

Output:
- results_naive_benchmarks.csv

Usage in CLI:
    python naive_benchmarks/naive_benchmarks.py
"""

import numpy as np
import pandas as pd
import os

# read the dataframe
df = pd.read_csv("data_intermediate/transformed_multivariate_domain_1.csv")

# skip the transformation indicator row
df = df.iloc[1:,]

df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate")

# select only  INDPRO, as naive forecasts solely based on INDPRO are computed
df = df[["INDPRO"]]

# setting the evaluation time frame - as with the other models
# filter from 1st July 1976 to and including September 1st, 2025 (evaluation time frame) - for consistency reasons in this study
evaluation = df.loc['1976-07-01':'2025-09-01'].copy()

y_true_all = evaluation["INDPRO"].values

# definition of the economic sub-regimes
REGIMES = {
    "pre-GFC": ("1900-01-01", "2007-07-31"),  
    "GFC": ("2007-08-01", "2009-06-30"),
    "post-GFC/pre-COVID": ("2009-07-01", "2020-02-29"),
    "COVID": ("2020-03-01", "2023-04-30"),
    "post-COVID": ("2023-05-01", "2100-01-01")
}

# metric computation function (RMSE, MAE, Hit Rate)
def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    return rmse, mae, hit_rate

# evaluation function
def evaluate_model(name, y_pred_series):
    results = []
    y_true_all = evaluation["INDPRO"]
    aligned = pd.concat([y_true_all, y_pred_series], axis=1).dropna()
    y_true = aligned.iloc[:, 0].values
    y_pred = aligned.iloc[:, 1].values
    rmse_full, mae_full, hit_full = compute_metrics(y_true, y_pred)
    results.extend([rmse_full, mae_full, hit_full])
    for regime_name, (start, end) in REGIMES.items():
        regime_data = aligned.loc[start:end]
        if len(regime_data) == 0:
            results.extend([np.nan, np.nan, np.nan])
            continue
        y_true_r = regime_data.iloc[:, 0].values
        y_pred_r = regime_data.iloc[:, 1].values
        rmse, mae, hit = compute_metrics(y_true_r, y_pred_r)
        results.extend([rmse, mae, hit])
    return [name] + results

# three different naive benchmarks
# constant zero
y_zero = pd.Series(0, index=evaluation.index)

# naive persistence
y_persistence = evaluation["INDPRO"].shift(1)

# expanding mean
y_expanding_mean = evaluation["INDPRO"].expanding().mean().shift(1)

# specifying the columns - consistent with the model forecast summary CSVs
columns = [
    "Naive_Benchmark",
    "RMSE_full", "MAE_full", "HitRate_full",
    "RMSE_pre-GFC", "MAE_pre-GFC", "HitRate_pre-GFC",
    "RMSE_GFC", "MAE_GFC", "HitRate_GFC",
    "RMSE_post-GFC/pre-COVID", "MAE_post-GFC/pre-COVID", "HitRate_post-GFC/pre-COVID",
    "RMSE_COVID", "MAE_COVID", "HitRate_COVID",
    "RMSE_post-COVID", "MAE_post-COVID", "HitRate_post-COVID"
]

results_all = []

results_all.append(evaluate_model("Naive_Zero", y_zero))
results_all.append(evaluate_model("Naive_Persistence", y_persistence))
results_all.append(evaluate_model("Naive_RollingMean_12", y_expanding_mean))

output_df = pd.DataFrame(results_all, columns=columns)

output_df.to_csv("./results/forecast/results_naive_benchmarks.csv", index=False)

# open CSV file
#os.startfile(r"naive_benchmarks\results_naive_benchmarks.csv")


