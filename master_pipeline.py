import subprocess
import sys
from pathlib import Path
import os

# ensure that all folders for storing the intermediate data and results do exist - a new folder is created if it does not exist yet
os.makedirs("data_intermediate", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/descriptive", exist_ok=True)
os.makedirs("results/forecast", exist_ok=True)
os.makedirs("results/forecast/visualizations", exist_ok=True)
os.makedirs("results/shap", exist_ok=True)
os.makedirs("results/shap/visualizations", exist_ok=True)


def run_step(description, command):
    print(f"\nRunning {description}...")
    subprocess.run(command, check=True)


print("Running univariate pipelines")
# start of with all univariate pipelines
run_step(
    "Elastic Net univariate pipeline",
    [sys.executable, "pipelines/elasticnet_univariate_pipeline.py"]
)

run_step(
    "Huber Regression univariate pipeline",
    [sys.executable, "pipelines/huber_univariate_pipeline.py"]
)

run_step(
    "LightGBM univariate pipeline",
    [sys.executable, "pipelines/lightgbm_univariate_pipeline.py"]
)

run_step(
    "MLP univariate pipeline",
    [sys.executable, "pipelines/mlp_univariate_pipeline.py"]
)

run_step(
    "SGD Regression univariate pipeline",
    [sys.executable, "pipelines/sgd_univariate_pipeline.py"]
)


print("Running filter-based multivariate pipelines")
# then all filter-based multivariate pipelines
run_step(
    "ElasticNet multivariate pipeline",
    [sys.executable, "pipelines/elasticnet_multivariate_pipeline.py"]
)

run_step(
    "Huber multivariate pipeline",
    [sys.executable, "pipelines/huber_multivariate_pipeline.py"]
)

run_step(
    "LightGBM multivariate pipeline",
    [sys.executable, "pipelines/lightgbm_multivariate_pipeline.py"]
)

run_step(
    "MLP multivariate pipeline",
    [sys.executable, "pipelines/mlp_multivariate_pipeline.py"]
)

run_step(
    "SGD multivariate pipeline",
    [sys.executable, "pipelines/sgd_multivariate_pipeline.py"]
)


print("Finally all domain-informed multivariate pipelines")
# then all domain-informed multivariate pipelines
run_step(
    "ElasticNet multivariate domain pipeline",
    [sys.executable, "pipelines/elasticnet_multivariate_domain_pipeline.py"]
)

run_step(
    "Huber multivariate domain pipeline",
    [sys.executable, "pipelines/huber_multivariate_domain_pipeline.py"]
)

run_step(
    "LightGBM multivariate domain pipeline",
    [sys.executable, "pipelines/lightgbm_multivariate_domain_pipeline.py"]
)

run_step(
    "MLP multivariate domain pipeline",
    [sys.executable, "pipelines/mlp_multivariate_domain_pipeline.py"]
)

run_step(
    "SGD multivariate domain pipeline",
    [sys.executable, "pipelines/sgd_multivariate_domain_pipeline.py"]
)

print("Running naive benchmarks")
run_step(
    "Naive benchmark computation",
    [sys.executable, "naive_benchmarks/naive_benchmarks.py"]
)

print("Running overlay visualization of univariate models")
run_step(
    "Univariate overlay visualization",
    [
        sys.executable,
        "visualization/forecast_visualization_overlay.py",
        "--type",
        "univariate"
    ]
)

print("Running overlay visualization of multivariate models")
run_step(
    "Multivariate overlay visualization",
    [
        sys.executable,
        "visualization/forecast_visualization_overlay.py",
        "--type",
        "multivariate"
    ]
)

print("Running overlay visualization of domain-informed multivariate models")
run_step(
    "Domain-informed multivariate overlay visualization",
    [
        sys.executable,
        "visualization/forecast_visualization_overlay.py",
        "--type",
        "DI-multivariate"
    ]
)

print("\nMaster pipeline finished successfully. For further insights, please inspect the artifacts.")