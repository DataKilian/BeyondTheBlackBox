"""
This script creates descriptive statistics and visualizations of INDPRO in the raw and differenced state.
The summary statistics are saved in a CSV file.


Output:
- description_raw.csv
- description_transformed.csv

- indpro_raw_plot.svg (line plot)
- indpro_transformed_plot.svg (line plot)
- indpro_raw_plot_reg.svg (line plot with indicated regimes)
- indpro_transformed_plot_reg.svg (line plot with indicated regimes)
- indpro_raw_hist.svg (histogram plot)
- indpro_transformed_hist.svg (histogram plot)
- indpro_raw_acf.svg (ACF plot)
- indpro_transformed_acf.svg (ACF plot)
- indpro_raw_pacf.svg (PACF plot)
- indpro_transformed_pacf.svg (PACF plot)


Usage in CLI:
    python analysis/descriptive_analysis.py --raw data_intermediate/screened_univariate.csv --transformed data_intermediate/transformed_univariate_1.csv --datecol sasdate
    
    The following two commands are redundant, as this script only computes statistics and creates visualizations for the INDPRO variable, which is the same series in all input datasets of each model type (univariate, multivariate, multivariate domain).
    
    python analysis/descriptive_analysis.py --raw data_intermediate/screened_multivariate.csv --transformed data_intermediate/transformed_multivariate_1.csv --datecol sasdate
    python analysis/descriptive_analysis.py --raw data_intermediate/screened_multivariate_domain.csv --transformed data_intermediate/transformed_multivariate_domain_1.csv --datecol sasdate

"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from scipy.stats import norm

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif" # use serif to match latex type style for the thesis

REGIMES = { # evaluation regimes defined, fur further inormation refer to paper
    "Pre-GFC": ("1900-01-01", "2007-07-31"),  
    "GFC": ("2007-08-01", "2009-06-30"),
    "Post-GFC Recovery": ("2009-07-01", "2020-02-29"),
    "SARS-CoV-2 Pandemic": ("2020-03-01", "2023-04-30"),
    "Post-Pandemic Era": ("2023-05-01", "2100-01-01")
}


# color palette for visualisation of sub-regimes
REGIME_COLORS = {
    "Pre-GFC": "#315459",            
    "GFC": "#7b7c6c",                  
    "Post-GFC Recovery": "#b2b09f", 
    "SARS-CoV-2 Pandemic": "#919890",         
    "Post-Pandemic Era": "#566359" 
    }  


# function taking a pandas series as input and computing summary statistics
def descriptive_stats(series: pd.Series):
    return {
        "count": series.count(),
        "mean": series.mean(),
        "std": series.std(),
        "varcof": series.std() / series.mean(),
        "min": series.min(),
        "25%": series.quantile(0.25),
        "50%": series.median(),
        "75%": series.quantile(0.75),
        "max": series.max(),
        "variance": series.var(),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis()
    }

# function to process the time series - calls other helper functions
def process_series(df: pd.DataFrame, label: str, date_col: str):
    
    # ensure the target variable "INDPRO" is available in the datafrmae
    if 'INDPRO' not in df.columns:
        raise ValueError(f"Time Series 'INDPRO' not found in {label} dataset.")
    
    # check if the date column is available
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in {label} dataset.")
    
    # get rid of the transform indicator row, to not distort summary statistics
    if df.iloc[0][date_col].startswith("Transform"):
        df = df.iloc[1:].copy()
    
       
    # ensure numeric INDPRO
    series = pd.to_numeric(df['INDPRO'], errors='coerce')
    series.index = pd.DatetimeIndex(df[date_col].values)
    series = series.dropna()

    # calculate regime statistics through the function defined below
    regime_df = regime_stats(series, REGIMES)

    # print regime statistics
    print(f"\n{label.upper()} INDPRO – Regime-specific mean and std:")
    print(regime_df[["mean", "std"]])
    
    # save the sescriptive statistics in a csv and corresponding folder
    stats = descriptive_stats(series)
    pd.DataFrame(stats, index=[label]).T.to_csv(f"results/descriptive/description_{label}.csv")
    
    # line plot with dates
    plt.figure()
    if label == "raw":
        series.plot(title='Industrial Production', color="#333333")
    if label == "transformed":
        series.plot(title='Monthly Industrial Production Growth', color="#333333")
    plt.xlabel(None)
    plt.ylabel('INDPRO' if label=='raw' else 'Δlog(INDPRO)')
    plt.tight_layout() 
    plt.savefig(f"results/descriptive/indpro_{label}_plot.svg", format='svg')
    plt.close()

    # line plot with color-indicated sub-regimes, as seen in the report
    fig, ax = plt.subplots()

    if label == "raw":
        series.plot(ax=ax, color="#333333", title="Industrial Production")
    else:
        series.plot(ax=ax, color="#333333", title="Monthly Industrial Production Growth")

    # create a regime shading
    regime_handles = []
    for regime, (start, end) in REGIMES.items():
        h = ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color=REGIME_COLORS[regime],
            alpha=0.45,
            linewidth=0
        )
        regime_handles.append(h)

    # set the labels
    ax.set_xlabel(None)
    ax.set_ylabel("INDPRO" if label == "raw" else "Δlog(INDPRO)")

    # include a legend indicating the regimes
    ax.legend(
        regime_handles,
        REGIMES.keys(),
        frameon=True,
        facecolor="white",
        edgecolor ="none",
        fontsize=9,
        loc="lower left"
    )

    # save the plot
    plt.tight_layout()
    plt.savefig(f"results/descriptive/indpro_{label}_plot_reg.svg", format="svg")
    plt.close()

    # create a histogram to inspect distribution
    plt.figure()
    sns.histplot(series, bins=30, kde=True, color="#333333")
    # fit normal distribution for comparison
    mu, sigma = series.mean(), series.std()
    x = np.linspace(series.min(), series.max(), 1000)
    # compute correct bin width from histogram
    counts, bins = np.histogram(series, bins=30)
    bin_width = bins[1] - bins[0]
    # plot scaled normal distribution PDF (to match the frequency dimension)
    plt.plot(x, norm.pdf(x, mu, sigma) * len(series) * bin_width, color="red", linestyle="-", linewidth=2, label="Normal Distribution")
    if label == "raw":
        plt.title("Distribution of Industrial Production")
    if label == "transformed":
        plt.title("Distribution of Monthly Industrial Production Growth")
    plt.xlabel('INDPRO' if label=='raw' else 'Δlog(INDPRO)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/descriptive/indpro_{label}_hist.svg", format='svg')
    plt.close()
    
    # creating ACF / PACF plots only for transformed, as whole forecasting exercise is only performed on the transformed data series
    if label == 'transformed':
        plt.figure()
        plot_acf(series, lags=40, color="#333333")
        plt.title(f'ACF of Monthly Industrial Production Growth')
        plt.xlabel('Lag')
        plt.tight_layout()
        plt.savefig(f"results/descriptive/indpro_{label}_acf.svg", format='svg')
        plt.close()
        
        plt.figure()
        plot_pacf(series, lags=40, color="#333333")
        plt.title("PACF of Monthly Industrial Production Growth")
        plt.xlabel('Lag')
        plt.tight_layout()
        plt.savefig(f"results/descriptive/indpro_{label}_pacf.svg", format='svg')
        plt.close()
    
    return stats


# stats calculation function
def regime_stats(series: pd.Series, regimes: dict):
    rows = []
    for name, (start, end) in regimes.items():
        mask = (series.index >= pd.to_datetime(start)) & (series.index <= pd.to_datetime(end)) # create a mask to create subsets of the time series for partial evaluation
        sub = series.loc[mask]

        rows.append({
            "regime": name,
            "start": start,
            "end": end,
            "count": sub.count(),
            "mean": sub.mean(),
            "std": sub.std()
        })

    return pd.DataFrame(rows).set_index("regime")


# main function, parsing arguments and calling helper functions
def main():
    parser = argparse.ArgumentParser() # create an argument parser instance to process inputs through CLI
    parser.add_argument("--raw", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--transformed", type=str, required=True, help="Path to transformed CSV")
    parser.add_argument("--datecol", type=str, default=None, help="Name of the date column")
    args = parser.parse_args()
    
    # load both datasets
    df_raw = pd.read_csv(args.raw)
    df_transformed = pd.read_csv(args.transformed)
    
    # drop first row (transformation indicators)
    df_raw = df_raw.iloc[1:].reset_index(drop=True)
    df_transformed = df_transformed.iloc[1:].reset_index(drop=True)
    
    # auto-detect date column if not provided (assume first column - as typical in time series datasets)
    date_col = args.datecol or df_raw.columns[0]
    
    stats_raw = process_series(df_raw, 'raw', date_col)
    stats_transformed = process_series(df_transformed, 'transformed', date_col)
    
    print("\nDescriptive statistics and SVG plots saved in 'results/descriptive/' folder.")
    print("\nRaw INDPRO stats:", stats_raw) # print stats
    print("\nTransformed INDPRO stats:", stats_transformed) # print stats

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    main()
