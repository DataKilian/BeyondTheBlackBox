import subprocess
import sys
from pathlib import Path

# move to project root in order for all path specification to work
project_root = Path(__file__).resolve().parent.parent
import os
os.chdir(project_root)

def run_step(description, command):
    print(f"\nRunning {description}...")
    subprocess.run(command, check=True)

run_step(
    "variable selection",
    [
        sys.executable,
        "data_handling/variable_selection.py",
        "--input", "input/2025-12-MD.csv",
        "--output", "data_intermediate/screened_univariate.csv",
        "--target", "INDPRO",
        "--type", "univariate"
    ]
)

run_step(
    "data handling",
    [
        sys.executable,
        "data_handling/variable_transformation_1.py",
        "--input", "data_intermediate/screened_univariate.csv",
        "--output", "data_intermediate/transformed_univariate_1.csv"
    ]
)

run_step(
    "stationarity analysis",
    [
        sys.executable,
        "analysis/stationarity_analysis.py",
        "--input", "data_intermediate/transformed_univariate_1.csv",
        "--output", "results/descriptive/stationarity_univariate_1.csv"
    ]
)

run_step(
    "descriptive analysis",
    [
        sys.executable,
        "analysis/descriptive_analysis.py",
        "--raw", "data_intermediate/screened_univariate.csv",
        "--transformed", "data_intermediate/transformed_univariate_1.csv",
        "--datecol", "sasdate"
    ]
)

run_step(
    "modeling",
    [
        sys.executable,
        "univariate_models/lightgbm_univariate.py",
        "--input", "data_intermediate/transformed_univariate_1.csv",
        "--datecol", "sasdate",
        "--start-train-months", "100"
    ]
)

run_step(
    "forecast visualization",
    [
        sys.executable,
        "visualization/forecast_visualization.py",
        "--input", "results/forecast/1_lightgbm_univariate_forecasts.csv",
        "--model", "Univariate LightGBM"
    ]
)

run_step(
    "SHAP overall visualization",
    [
        sys.executable,
        "visualization/shap_visualization_overall.py",
        "--input", "results/shap/1_lightgbm_univariate_shap_values.csv"
    ]
)

run_step(
    "SHAP dependence visualization",
    [
        sys.executable,
        "visualization/shap_visualization_dependence.py",
        "--input", "results/shap/1_lightgbm_univariate_shap_values.csv"
    ]
)

print("\nPipeline finished successfully. For further insights, please inspect the artifacts.")