"""
This script overlays the forecasts of all models within a specific model type (univariate, multivariate, DI-multivariate) and saves the figure as Scalable Vector Graphic (SVG).

Input:
- folder: ./results/forecast/
- CSV files following naming scheme:
    1_<model>_<model_type>_forecasts.csv

Output:
- {model_type}_actual_vs_all.svg (actual INDPRO growth vs. all model forecasts)
- {model_type}_all_deviations.svg (deviations of all model predictions from the actual INDPRO growth)

Usage in CLI:

    python visualization/forecast_visualization_overlay.py --type univariate
    python visualization/forecast_visualization_overlay.py --type multivariate
    python visualization/forecast_visualization_overlay.py --type DI-multivariate
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# matplotlib configuration for consistency with other visualization scripts and visually fitting integration into the thesis
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

ACTUAL_COLOR = "#222222"

# defining a well-distinguishable color palette
MODEL_COLORS = [
    "#2E5984", 
    "#3C9D9B", 
    "#A63D40", 
    "#7F5F9B", 
    "#C9A066",  
]

# utility function to collect all the forecast CSVs belonging to the specific model type (univariate/multivariate/DI-multivariate)
def get_model_files(model_type):
    base_path = "results/forecast"
    
    if model_type == "univariate":
        pattern = "*_univariate_forecasts.csv"
    elif model_type == "multivariate":
        pattern = "*_multivariate_forecasts.csv"
    elif model_type == "DI-multivariate":
        pattern = "*_multivariate_domain_forecasts.csv"
    else:
        raise ValueError("Type must be one of: univariate, multivariate, DI-multivariate")

    return sorted(glob.glob(os.path.join(base_path, pattern)))

# plotting function to visualize the overlay of forecasts vs the actual INDPRO growth
def plot_actual_vs_all(models_data, output_svg, model_type):
    fig, ax = plt.subplots(figsize=(10, 4))

    # plot actual INDPRO growth first
    first_df = list(models_data.values())[0]
    ax.plot(
        first_df["forecast_date"],
        first_df["actual_logdiff"],
        label="Actual",
        color=ACTUAL_COLOR,
        linewidth=1.8
    )

    # plot all model forecasts
    for i, (model_name, df) in enumerate(models_data.items()):
        ax.plot(
            df["forecast_date"],
            df["forecast_logdiff"],
            label=model_name,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            linewidth=1.2,
            alpha=0.9
        )

    ax.set_title(f"Monthly INDPRO Growth: Actual vs Forecasts ({model_type.capitalize()})", fontsize=11)
    ax.set_ylabel("Δlog(INDPRO)")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved overlay forecast plot to {output_svg}")

# plotting function to visualize the overlay of deviations of model forecasts vs the actual INDPRO growth
def plot_all_deviations(models_data, output_svg, model_type):
    fig, ax = plt.subplots(figsize=(10, 3))

    for i, (model_name, df) in enumerate(models_data.items()):
        df["deviation"] = df["actual_logdiff"] - df["forecast_logdiff"]

        ax.plot(
            df["forecast_date"],
            df["deviation"],
            label=model_name,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            linewidth=1.1,
            alpha=0.9
        )

    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)

    ax.set_title(f"Monthly INDPRO Growth: Forecast Deviations ({model_type.capitalize()})", fontsize=11)
    ax.set_ylabel("Δlog(INDPRO)")
    ax.legend(frameon=False, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved deviation overlay plot to {output_svg}")

# aggregate function - main routine
def visualize_type(model_type):
    # collect the model files (list of paths)
    model_files = get_model_files(model_type)

    if not model_files:
        raise FileNotFoundError(f"No forecast files found for type '{model_type}'")

    models_data = {}

    # create tidy and consistent legend names
    MODEL_NAME_MAP = {
        "elasticnet": "Elastic Net",
        "huber": "Huber Regression",
        "sgd": "SGD Regression",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
    }

    for file in model_files:
        filename = os.path.basename(file).lower()

        model_key = None
        for key in MODEL_NAME_MAP.keys():
            if key in filename:
                model_key = key
                break

        if model_key is None:
            continue  # skip unknown files safely

        model_name = MODEL_NAME_MAP[model_key]

         # Drop SGD for DI-Multivariate due to the extreme outliers
        if model_type == "DI-multivariate" and model_name == "SGD Regression":
            continue


        df = pd.read_csv(file, parse_dates=["forecast_date"])
        df = df.sort_values("forecast_date")

        models_data[model_name] = df

    output_dir = "results/forecast/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    plot_actual_vs_all(
        models_data,
        f"{output_dir}/{model_type}_actual_vs_all.svg",
        model_type
    )

    plot_all_deviations(
        models_data,
        f"{output_dir}/{model_type}_all_deviations.svg",
        model_type
    )


# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay all model forecasts for a specific type"
    )

    parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=["univariate", "multivariate", "DI-multivariate"],
        help="Model type to visualize"
    )

    args = parser.parse_args()
    visualize_type(args.model_type)