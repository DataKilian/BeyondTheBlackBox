"""
This script creates single visualizations on the forecasting performance (fit and deviation) of a selected model and saves them as a Scalable Vector Graphic (SVG). 

Input:
- folder: ./results/forecast/
- CSV files following naming scheme:
    1_<model_class>_<model_type>_forecasts.csv

From the input file, the following columns are retrieved: forecast_date, actual_logdiff, forecast_logdiff

Output:
- {output_prefix}_actual_vs_forecast.svg (actual INDPRO growth vs. forecasts of the model)
- {output_prefix}_deviation.svg (deviation of forecasts from the actual INDPRO growth)

The {output_prefix} consists of {forecasting_horizon}_{model_type}_{model_class}
    
Usag in CLI:

    python visualization/forecast_visualization.py --input results/forecast/1_elasticnet_univariate_forecasts.csv --model "Univariate Elastic Net"
    python visualization/forecast_visualization.py --input results/forecast/1_huber_univariate_forecasts.csv --model "Univariate Huber Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_sgd_univariate_forecasts.csv --model "Univariate SGD Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_lightgbm_univariate_forecasts.csv --model "Univariate LightGBM"
    python visualization/forecast_visualization.py --input results/forecast/1_mlp_univariate_forecasts.csv --model "Univariate MLP"

    python visualization/forecast_visualization.py --input results/forecast/1_elasticnet_multivariate_forecasts.csv --model "Multivariate Elastic Net"
    python visualization/forecast_visualization.py --input results/forecast/1_huber_multivariate_forecasts.csv --model "Multivariate Huber Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_sgd_multivariate_forecasts.csv --model "Multivariate SGD Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_lightgbm_multivariate_forecasts.csv --model "Multivariate LightGBM"
    python visualization/forecast_visualization.py --input results/forecast/1_mlp_univariate_forecasts.csv --model "Multivariate MLP"

    python visualization/forecast_visualization.py --input results/forecast/1_elasticnet_multivariate_domain_forecasts.csv --model "DI-Multivariate Elastic Net"
    python visualization/forecast_visualization.py --input results/forecast/1_huber_multivariate_domain_forecasts.csv --model "DI-Multivariate Huber Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_sgd_multivariate_domain_forecasts.csv --model "DI-Multivariate SGD Regression"
    python visualization/forecast_visualization.py --input results/forecast/1_lightgbm_multivariate_domain_forecasts.csv --model "DI-Multivariate LightGBM"
    python visualization/forecast_visualization.py --input results/forecast/1_mlp_multivariate_domain_forecasts.csv --model "DI-Multivariate MLP"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# matplotlib configuration for consistency with other visualization scripts and fitting integration into the thesis
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

ACTUAL_COLOR = "#333333"
FORECAST_COLOR = "#B0E0E6"
DEVIATION_COLOR = "#333333"

# plotting function to visualize forecasts vs the actual INDPRO growth
def plot_actual_vs_forecast(df, output_svg, model_name):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df["forecast_date"],
        df["actual_logdiff"],
        label="Actual",
        color=ACTUAL_COLOR,
        linewidth=1.5
    )

    ax.plot(
        df["forecast_date"],
        df["forecast_logdiff"],
        label="Forecast",
        color=FORECAST_COLOR,
        linewidth=1.5
    )

    ax.set_title(f"Monthly INDPRO Growth: Actual vs. Forecasted ({model_name})", fontsize=11)
    ax.set_xlabel(None)
    ax.set_ylabel("Δlog(INDPRO)")

    ax.legend(frameon=False)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved actual vs. forecast plot to {output_svg}")

# plotting function to visualize the deviation of model forecast vs the actual INDPRO growth
def plot_deviation(df, output_svg, model_name):
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        df["forecast_date"],
        df["deviation"],
        color=DEVIATION_COLOR,
        linewidth=1.2
    )

    ax.axhline(
        0,
        color=FORECAST_COLOR,
        linewidth=1.0,
        linestyle="--"
    )

    ax.set_title(f"Monthly INDPRO Growth: Forecast Deviation ({model_name})", fontsize=11)
    ax.set_xlabel(None)
    ax.set_ylabel("Δlog(INDPRO)")

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved deviation plot to {output_svg}")

# aggregate function - main routine
def visualize_results(input_csv, model_name):
    base_name = os.path.basename(input_csv)
    if base_name.endswith("_forecasts.csv"):
        output_prefix = base_name.replace("_forecasts.csv", "")
    else:
        output_prefix = os.path.splitext(base_name)[0]

    df = pd.read_csv(
        input_csv,
        parse_dates=["forecast_date"]
    ).sort_values("forecast_date")

    df["deviation"] = df["actual_logdiff"] - df["forecast_logdiff"]

    plot_actual_vs_forecast(
        df,
        f"results/forecast/visualizations/{output_prefix}_actual_vs_forecast.svg", model_name
    )

    plot_deviation(
        df,
        f"results/forecast/visualizations/{output_prefix}_deviation.svg", model_name
    )

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot actual vs forecast values and deviation from forecasts"
    )
    parser.add_argument(
        "--input",
        required=True
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of model for plotting"
    )


    args = parser.parse_args()
    visualize_results(input_csv = args.input, model_name = args.model)
