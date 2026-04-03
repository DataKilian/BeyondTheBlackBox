"""
This script creates visualizations of global SHAP values for the specified model and saves them as a SVG.

Input:
- folder: ./results/shap/
- CSV files following naming scheme:
    1_<model_class>_<model_type>_shap_values.csv

- CSV file using columns:
From the input file, the following columns are retrieved: origin_date, step, feature, shap_value, feature_value

Output:
- {output_prefix}_global_shap_c.svg (global absolute feature importance (bar plot))
- {output_prefix}_global_shap_beeswarm_c.svg (SHAP beeswarm plot (per-feature distribution))

The {output_prefix} consists of {forecasting_horizon}_{model_class}_{model_type}

A description and explanation of the figures can be found in the thesis.

Usage in CLI:
    python visualization/shap_visualization_overall.py --input results/shap/1_elasticnet_univariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_huber_univariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_sgd_univariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_lightgbm_univariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_mlp_univariate_shap_values.csv

    python visualization/shap_visualization_overall.py --input results/shap/1_elasticnet_multivariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_huber_multivariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_sgd_multivariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_lightgbm_multivariate_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_mlp_multivariate_shap_values.csv

    python visualization/shap_visualization_overall.py --input results/shap/1_elasticnet_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_huber_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_sgd_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_lightgbm_multivariate_domain_shap_values.csv
    python visualization/shap_visualization_overall.py --input results/shap/1_mlp_multivariate_domain_shap_values.csv
"""


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
import shap

# matplotlib configuration for consistency with other visualization scripts and visually fitting integration into the thesis
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

BAR_COLOR = "#333333"

# handle model names
def parse_model_display_name(filename):

    filename = filename.lower()

    # determine model type
    if "multivariate_domain" in filename:
        model_class = "DI-Multivariate"
    elif "multivariate" in filename:
        model_class = "Multivariate"
    elif "univariate" in filename:
        model_class = "Univariate"
    else:
        model_class = ""

    # determine model class
    if "elasticnet" in filename:
        model_name = "Elastic Net"
    elif "huber" in filename:
        model_name = "Huber Regression"
    elif "sgd" in filename:
        model_name = "SGD Regression"
    elif "lightgbm" in filename:
        model_name = "LightGBM"
    elif "mlp" in filename:
        model_name = "MLP"
    else:
        model_name = ""

    if model_class and model_name:
        return f"{model_class} {model_name}"
    else:
        return model_name

# function to plot the global average absolute SHAP values in a bar plot
def plot_global_shap_bar(df, output_prefix, display_name):
    global_shap = (
        df.groupby("feature")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(global_shap) * 0.4)))

    ax.barh(global_shap.index, global_shap.values, color=BAR_COLOR, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance ({display_name})", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    output_svg = f"results/shap/visualizations/{output_prefix}_global_shap_c.svg"
    plt.savefig(output_svg, format="svg")
    plt.close()
    print(f"Saved global SHAP bar plot to {output_svg}")

# function to plot a beeswarm plot of global SHAP to investigate distribution
# the per-built SHAP beeswarm plotting function from the SHAP package is used
def plot_shap_beeswarm(df, output_prefix, display_name):

    duplicates = df.groupby(['origin_date', 'feature']).size()
    print(duplicates[duplicates > 1])
    features = df["feature"].unique()
    origin_dates = sorted(df["origin_date"].unique())
    X_matrix = pd.DataFrame(index=origin_dates, columns=features)
    shap_matrix = pd.DataFrame(index=origin_dates, columns=features)

    for date in origin_dates:
        temp = df[df["origin_date"] == date]
        X_matrix.loc[date, temp["feature"]] = temp["feature_value"].values
        shap_matrix.loc[date, temp["feature"]] = temp["shap_value"].values

    X_matrix = X_matrix.astype(float).values
    shap_matrix = shap_matrix.astype(float).values

    # SHAP explanation object
    explainer = shap.Explanation(values=shap_matrix, data=X_matrix, feature_names=features)

    # max_display attribute would allow to restrict the number of visualized features - however, to gain a complete picture, all features are included
    fig = plt.figure(figsize=(6, max(4, len(features) * 0.4)))


    shap.plots.beeswarm(explainer, show=False, max_display = 28)
    plt.xlabel("SHAP value")  # <-- Set x-axis label
    plt.title(f"SHAP Beeswarm ({display_name})", fontsize=12)
    plt.tight_layout()
    output_svg = f"results/shap/visualizations/{output_prefix}_global_shap_beeswarm_c.svg"
    plt.savefig(output_svg, format="svg")
    plt.close()
    print(f"Saved SHAP beeswarm plot to {output_svg}")



# main orchestrating function for visualizing global SHAP values
def visualize_global_shap(input_csv):
    base_name = os.path.basename(input_csv)
    if base_name.endswith("_shap_values_direct.csv"):
        output_prefix = base_name.replace("_shap_values_direct.csv", "")
    else:
        output_prefix = os.path.splitext(base_name)[0]

    display_name = parse_model_display_name(base_name)

    df = pd.read_csv(input_csv)

    plot_global_shap_bar(df, output_prefix, display_name)
    plot_shap_beeswarm(df, output_prefix, display_name)

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot global SHAP values (bar plot & beeswarm) from forecasts"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SHAP CSV file"
    )

    args = parser.parse_args()
    visualize_global_shap(args.input)
