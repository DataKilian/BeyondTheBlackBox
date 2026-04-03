"""
This script creates visualizations of SHAP dependence (SHAP dependece plots) for every variable in the feature matrix of a specific model and saves them as a SVG in a dedicated folder.

Input:
- folder: ./results/shap/
- CSV files following naming scheme:
    1_<model_class>_<model_type>_shap_values.csv

From the input file, the following columns are retrieved: feature, feature_value, shap_value

Output (one SVG figure per variable):
- {output_prefix}_shap_dependence_{feature_name}.svg

The {output_prefix} consists of {forecasting_horizon}_{model_class}_{model_type}_shap_values

In the figure, SHAP values (y-axis) vs observed feature value (x-axis) are plotted. A 3rd-degree polynomial, fit to facilitate inspecting the functional form, is overlayed.

Usage in CLI:

    python visualization/shap_visualization_dependence.py --input results/shap/1_elasticnet_univariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_huber_univariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_sgd_univariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_lightgbm_univariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_mlp_univariate_shap_values.csv

    python visualization/shap_visualization_dependence.py --input results/shap/1_elasticnet_multivariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_huber_multivariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_sgd_multivariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_lightgbm_multivariate_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_mlp_multivariate_shap_values.csv

    python visualization/shap_visualization_dependence.py --input results/shap/1_elasticnet_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_huber_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_sgd_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_lightgbm_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_dependence.py --input results/shap/1_mlp_multivariate_domain_shap_values.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

# matplotlib configuration for consistency with other visualization scripts and visually fitting integration into the thesis
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

POINT_SIZE = 28
POINT_ALPHA = 0.7
LINE_WIDTH = 2
FIT_LINE_COLOR = "#333333"  # polynomial fit line
ZERO_LINE_COLOR = "#333333"  # zero line

# colors set for the models - to be consistent with other visualization
MODEL_COLORS = [
    "#2E5984",  
    "#3C9D9B",  
    "#A63D40", 
    "#7F5F9B",  
    "#C9A066",  
]

MODEL_NAMES = ["elasticnet", "huber", "lightgbm", "mlp", "sgd"]


# retrieve the color based on the model name
def get_model_color(csv_path):
    fname = os.path.basename(csv_path).lower()
    for model_name, color in zip(MODEL_NAMES, MODEL_COLORS):
        if model_name in fname:
            return color
    return "#333333"  # fallback

# plotting function for SHAP dependence of a single input feature
def plot_shap_dependence(df, feature_name, output_dir, output_prefix, point_color):

    g = df[df["feature"] == feature_name]

    x = pd.to_numeric(g["feature_value"], errors="coerce").values
    y = pd.to_numeric(g["shap_value"], errors="coerce").values

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        color=point_color,
        alpha=POINT_ALPHA,
        linewidth=0
    )

    ax.axhline(0, linestyle="--", linewidth=1, color=ZERO_LINE_COLOR)

    # fit a 3rd degreee polinomial
    if len(x) >= 4:
        coeffs = np.polyfit(x, y, deg=3)
        poly = np.poly1d(coeffs)

        x_grid = np.linspace(x.min(), x.max(), 400)
        y_fit = poly(x_grid)

        ax.plot(
            x_grid,
            y_fit,
            color=FIT_LINE_COLOR,
            linewidth=LINE_WIDTH
        )

    # remove everything after '_' in feature names for tidy labels
    display_feature_name = feature_name.split("_")[0]

    ax.set_xlabel(f"{display_feature_name} (observed value)")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP dependence: {display_feature_name}", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    output_svg = output_dir / f"{output_prefix}_shap_dependence_{feature_name}.svg"
    plt.savefig(output_svg, format="svg")
    plt.close()

    print(f"Saved SHAP dependence plot to {output_svg}")


# main orchestrating function for visualizing SHAP dependence
def visualize_shap_dependence(input_csv):

    base_name = os.path.basename(input_csv)
    output_prefix = os.path.splitext(base_name)[0]

    # output directory per model in the results/shap folder
    model_name = [m for m in MODEL_NAMES if m in base_name.lower()]
    if model_name:
        output_dir = Path(f"results/shap/{model_name[0]}")
    else:
        output_dir = Path("results/shap/other")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = {"feature", "feature_value", "shap_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = sorted(df["feature"].unique())
    print(f"Creating SHAP dependence plots for {len(features)} features...")

    point_color = get_model_color(input_csv)

    for feature_name in features:
        plot_shap_dependence(df, feature_name, output_dir, output_prefix, point_color)



# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot SHAP dependence plots with cubic polynomial fits"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SHAP CSV file"
    )

    args = parser.parse_args()
    visualize_shap_dependence(args.input)
