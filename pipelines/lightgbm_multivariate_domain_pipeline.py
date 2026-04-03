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
        "--output", "data_intermediate/screened_multivariate_domain.csv",
        "--target", "INDPRO",
        "--type", "multivariate_domain"
    ]
)

run_step(
    "data handling",
    [
        sys.executable,
        "data_handling/variable_transformation_1.py",
        "--input", "data_intermediate/screened_multivariate_domain.csv",
        "--output", "data_intermediate/transformed_multivariate_domain_1.csv"
    ]
)

run_step(
    "stationarity analysis",
    [
        sys.executable,
        "analysis/stationarity_analysis.py",
        "--input", "data_intermediate/transformed_multivariate_domain_1.csv",
        "--output", "results/descriptive/stationarity_multivariate_domain_1.csv"
    ]
)

run_step(
    "descriptive analysis",
    [
        sys.executable,
        "analysis/descriptive_analysis.py",
        "--raw", "data_intermediate/screened_multivariate_domain.csv",
        "--transformed", "data_intermediate/transformed_multivariate_domain_1.csv",
        "--datecol", "sasdate"
    ]
)

run_step(
    "modeling",
    [
        sys.executable,
        "multivariate_models/lightgbm_multivariate_domain.py",
        "--input", "data_intermediate/transformed_multivariate_domain_1.csv",
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
        "--input", "results/forecast/1_lightgbm_multivariate_domain_forecasts.csv",
        "--model", "DI-Multivariate LightGBM"
    ]
)

run_step(
    "SHAP overall visualization",
    [
        sys.executable,
        "visualization/shap_visualization_overall.py",
        "--input", "results/shap/1_lightgbm_multivariate_domain_shap_values.csv"
    ]
)

run_step(
    "SHAP dependence visualization",
    [
        sys.executable,
        "visualization/shap_visualization_dependence.py",
        "--input", "results/shap/1_lightgbm_multivariate_domain_shap_values.csv"
    ]
)

print("\nPipeline finished successfully. For further insights, please inspect the artifacts.")