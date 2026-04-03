import subprocess
import sys
from pathlib import Path
import os

# move to project root in order for all path specification to work
project_root = Path(__file__).resolve().parent.parent
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
        "--output", "data_intermediate/screened_multivariate.csv",
        "--target", "INDPRO",
        "--type", "multivariate"
    ]
)

run_step(
    "data handling",
    [
        sys.executable,
        "data_handling/variable_transformation_1.py",
        "--input", "data_intermediate/screened_multivariate.csv",
        "--output", "data_intermediate/transformed_multivariate_1.csv"
    ]
)

run_step(
    "stationarity analysis",
    [
        sys.executable,
        "analysis/stationarity_analysis.py",
        "--input", "data_intermediate/transformed_multivariate_1.csv",
        "--output", "results/descriptive/stationarity_multivariate_1.csv"
    ]
)

run_step(
    "descriptive analysis",
    [
        sys.executable,
        "analysis/descriptive_analysis.py",
        "--raw", "data_intermediate/screened_multivariate.csv",
        "--transformed", "data_intermediate/transformed_multivariate_1.csv",
        "--datecol", "sasdate"
    ]
)

run_step(
    "modeling",
    [
        sys.executable,
        "multivariate_models/mlp_multivariate.py",
        "--input", "data_intermediate/transformed_multivariate_1.csv",
        "--datecol", "sasdate",
        "--nlags", "12",
        "--predictor-lags", "1",
        "--start-train-months", "100",
        "--verbose"
    ]
)

run_step(
    "forecast visualization",
    [
        sys.executable,
        "visualization/forecast_visualization.py",
        "--input", "results/forecast/1_mlp_multivariate_forecasts.csv",
        "--model", "Multivariate MLP"
    ]
)

run_step(
    "SHAP overall visualization",
    [
        sys.executable,
        "visualization/shap_visualization_overall.py",
        "--input", "results/shap/1_mlp_multivariate_shap_values.csv"
    ]
)

run_step(
    "SHAP dependence visualization",
    [
        sys.executable,
        "visualization/shap_visualization_dependence.py",
        "--input", "results/shap/1_mlp_multivariate_shap_values.csv"
    ]
)

print("\nPipeline finished successfully. For further insights, please inspect the artifacts.")