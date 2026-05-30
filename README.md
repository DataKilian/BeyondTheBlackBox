# Beyond the Black Box: Gaining Insights Using xAI when Forecasting U.S. Industrial Production

This code repository contains the data and scripts used in the Bachelor Thesis *Beyond the Black Box: Gaining Insights Using xAI when Forecasting U.S. Industrial Production* by Kilian Schicho.

The thesis aims to address the following three research questions:
- Do classical non-linear machine learning and deep learning models outperform established linear statistical models in forecasting U.S. Industrial Production (*INDPRO*) growth in terms of predictive accuracy and computational efficiency?

- How can SHapley Additive exPlanations (SHAP) help uncover the economic structure behind the model predictions?

- Do different models produce consistent, economically meaningful explanations?

The data and code in this repository can be used to reproduce all results reported in the Bachelor Thesis.

In case a question arises, please send an e-mail to <kilian.schicho@s.wu.ac.at>.

## Abstract
The literature on economic time series forecasting has increasingly incorporated
advanced machine and deep learning models. While these model classes often improve predictive performance, model explainability issues emerged due to their complex architectures, often referred to as a ‘black boxes”. As a result, model transparency is limited and the economic logic behind predictions remains opaque.
To address this issue, this thesis provides empirical evidence on performance as
well as model explainability in a short-term forecasting setting of U.S. industrial
production. A range of linear statistical and non-linear machine learning models
are evaluated and compared across multiple forecasting setups. For each model
class, three model types are tested: univariate, domain-informed multivariate and
filter-based multivariate. The models’ performance is assessed multi-dimensionally,
by incorporating both computation efficiency and predictive accuracy.
First, an in-depth review of relevant literature was conducted to examine the
state of the art of time series forecasting and to investigate current research on
xAI in macroeconomic time series forecasting. Building on those insights, a multistage experimental pipeline for forecasting the one-month-ahead growth rate of U.S.
industrial production, was developed and implemented across various forecasting
setups.
The empirical analysis is based on the publicly available Federal Reserve Economic Data Monthly Database (FRED-MD), which represents a collection of 126
U.S. macroeconomic indicators. To simulate real-time forecasting conditions, forecasts are performed in a pseudo out-of-sample framework with an expanding window. SHapley Additive exPlanations (SHAP) value decompositions are performed
to deal with the explainability limitations of sophisticated models and to identify
key drivers of model predictions.
The central findings suggest that on average, non-linear machine learning models
tend to outperform their linear counterparts in terms of predictive accuracy, though
there are important distinctions. Specifically, LightGBM consistently demonstrates
superior performance. whereas the multi-layer perceptron (MLP) does not significantly outperform the tested linear models. While this finding holds across all model
types, LightGBM benefits particularly from the inclusion of additional variables in
the multivariate settings. Moreover, this thesis provides evidence that SHAP values are an effective tool for uncovering the economic structure underpinning model
predictions. In detail, the explainability analysis indicates, that the majority of
analyzed models’ decision patterns are economically meaningful, consistent across
model classes and in line with mainstream economic theory.
Overall, this thesis contributes to the macroeconomic forecasting literature by
jointly assessing performance and model explainability. The results highlights the
potential of forecasting using machine learning models and demonstrates the importance of SHAP values for understanding economic implications and assessing
model validity. Future research may further investigate the stability of explanation
patterns and further modifications of the experimental pipeline.

## Repository Overview
The repository includes:
- pipeline orchestration scripts for convenient reproducibility
- input data
- intermediate data
- data handling scripts
- analysis scripts
- modeling scripts
- visualization scripts
- final results

Each script starts with a docstring containing at least the following elements (if applicable):
1. Short description of the script and its purpose
2. Input data
3. Output data
4. Usage for Command Line Interface (CLI)

## Folder Structure
The repository is structured as follows:

```text
BeyondTheBlackBox_Kilian_Schicho/
│
├── README.md
│── README.html
├── requirements.txt
├── master_pipeline.py # master pipeline to run all separate model pipelines in sequence 
│
├── /input/ # FRED-MD (2025-12)
|
├── /intermediate_data/ # screened and transformed CSVs
|
├── /analysis/
│   ├── descriptive_analysis.py
│   └── stationarity_analysis.py
|
├── /data_handling/
│   ├── variable_selection.py
│   ├── variable_transformation_1.py
│   └── variable_transformation_3.py
|
├── /naive_benchmarks/ # naive benchmark script
│
├── /univariate_models/ # univariate model scripts and helper functions
│
├── /multivariate_models/ # multivariate model scripts and helper functions
│
|── results/
│   ├── /descriptive/ # plots and summary/stationarity statistic CSVs on raw and transformed dataset
│   ├── /forecast/ # forecast CSVs and visualizations
│   └── /shap/ # shap value CSV, beeswarm plots, global importance plots, SHAP dependence plots
│
├── /visualization/ # visualization scripts
│
└── /visualizations_paper/ # all visualizations included in the paper
│
└── /pipelines/ # all pipeline orchestration instances for efficiently executing the experimental pipeline per model
```

## Code Organization
This section provides a structured overview of the repository architecture and provides a guide on how to use the included scripts and pipelines. Moreover, it maps the stages of the experimental pipeline visualized in the thesis to scripts in this code repository.

### Data
There is one input dataset further described in section "Dataset" of the README file.

Intermediate datasets are automatically saved in the `/data_intermediate` folder. On the one hand, screened datasets as an output of the variable selection script, are stored here. Those CSV files include the variables selected for the specific model type: univariate, (filter-based) multivariate and domain-informed multivariate.

On the other hand, datasets containing the transformed variables (result of the transformation script), are saved here. Transformation have been applied based on one-month differences (default) and 3-month differences (benchmark).

### Data Handling
The files in the `/data_handling` folder implement the Data Loading,  Variable Selection and Data Preparation stages of the experimental pipeline. Among these, the script `variable_selection.py` covers the Data Loading and Variable Selection stage - all in one script. `variable_transformation_1.py` and `variable_transformation_3.py` contain the Data Preparation steps for the default setting (one-month) and the benchmark setting (three-months) respectively. As described in the thesis, Data Preparation is comprised of Data Cleaning (handling of missing values) and Variable Transformation (to induce stationarity).

### Analysis
The `/analysis` folder includes the stationary and descriptive analysis scripts. These scripts correspond to the Descriptive and Stationarity Analysis stage of the experimental pipeline. They are solely used to derive summary statistics and create visualizations. No modifications are made to the intermediate datasets in this stage.

### Feature Engineering, Modeling, Evaluation and Artifact Management
The Feature Engineering, Modeling, Evaluation and Artifact Management stages of the experimental pipline are implemented in the two different folders corresponding to model types investigated in the thesis: `/univariate_models` and `/multivariate_models`. The `/multivariate_models` folder contains scripts implementing both the filter-based multivariate as well as the domain-informed multivariate models. Each script name indicates a specific model type and model class combination. `elasticnet_multivariate_domain.py` for example, is the realization of the Elastic Net model class in the domain-informed multivariate model type setting. Furthermore, in each folder, a `model_helper.py` script can be found which comprises utility functions for all separate model scripts.

The scripts load the prepared datasets, perform feature engineering and execute the pseudo out-of-sample forecasting pipeline with an expanding window training methodology and an integrated explainability layer. The following sub-stages of the Modeling stage are implemented: Data Preprocessing (Standardization, Winsorization), Hyperparameter Tuning (every 12 months), Training, Prediction and SHAP Analysis. Subsequently, the predictions are evaluated across various metrics (+ computational intensity) and saved along with the forecasts. Additionally, SHAP values are collected and separately saved in a different CSV file. Finally, all optimal hyperparameter combinations are persisted in a third CSV file. All CSVs are saved in the `/results` folder. This corresponds to the Artifact Management stage of the pipeline.

### Naive Benchmarks
To provide a proper comparison baseline for the employed linear statistical, classical machine learning and deep learning models, naive benchmarks are evaluated in the `naive_benchmarks.py` script in the `/naive_benchmarks` folder. The resulting evaluation metrics are also saved in the `/results` folder.

### Visualization
Visualization scrips are collectively stored in the `/visualization` folder. `visualization_overlay.py` creates overlayed forecast plots of all models. `forecast_visualization.py` in contrast is used to visualize the forecasting paths of a single model.

On the explainability side, `shap_visualization_overall.py` creates SHAP beeswarm and global feature importance plots. The `shap_visualization_dependence.py` script visualizes SHAP dependence plots for each variable used in a specific model. These can be used to assess the economic relationships learned by the model, as demonstrated in the thesis.

In the `/visualization_paper` folder, all visualizations included in the thesis, are stored. In addition, the dependence folder also contains dependence plots showcasing interesting economic relationships, which were not included in the final version of the thesis.

### Results
The `/results` folder is composed of all resulting CSV files and visualizations of the experimental pipeline. It is structured into descriptive, forecast and SHAP explainability results.

An overview of the results found in each subfolder is given below:

`/descriptive`:
- summary statistics
- stationarity statistics
- histograms
- lineplots
- PACF plot
- ACF plot

`/forecast`:
- visualizations (forecast vs. actual, deviation)
- optimal hyperparameter files
- forecast files incl. evaluation metrics
- evaluation metrics of the naive benchmarks (`results_naive_benchmarks.csv`)

`/shap`:
- shap value files
- beeswarm plots
- global importance plots
- in each model class subfolder: all dependece plots for each model type

For a concise overview of the predictive performance of all models and naive benchmarks, please refer to `final_performance_comparison.xlsx`.

### Pipeline Orchestration Scripts
For convenience and enhanced reproducibility, pipeline orchestration scripts have been assembled in the `/pipelines` folder. Each script corresponds to a specific model instance. A total of 15 model instances (3 model types x 5 model classes) are implemented. When executed, a script runs all stages of the experimental pipeline for that model instance, including Data Loading, Variable Selection, Data Preparation, Stationarity Analysis, Descriptive Analysis, Feature Engineering, Modeling, Evaluation, Artifact Management and Visualization. 

The Model Selection stage is addressed inherently through the model classes implemented, as described and justified in detail in the thesis. 

To ensure a clear and consistent workflow across model instances, the execution order within the orchestration scripts is predefined.

Moreover, the file `master_pipeline.py` orchestrates the execution of all 15 model instances pipelines in sequence. Running this script therefore ensures that every pipeline instance described in the thesis is executed, for convenient and complete reproducibility. Additionally, as a last step, the naive benchmarks are computed. Thus, executing this file, runs all scripts in this repository, enabling the full reproduction of the experiments and analyses conducted in the thesis.

For further insights, the CLI commands used to execute each script and produce the results can be found in the docstring of each script. Furthermore, the output location is specified, along with details describing the purpose and structure of the script.

## Requirements and Dependencies
The experiment was conducted in Python 3.12.2, utilizing various external libraries. Those are listed in the `requirements.txt` file along with the version used.

The required libraries can be installed easily through executing the following command:

`pip install -r requirements.txt`

## Setup Used
The code repository was developed using Microsoft Visual Studio Code as an integrated development environment (IDE).

## Dataset
 The *FRED-MD* dataset (vintage of December 2025) was used as the only input data for the present experiment. It was obtained from the FRED Reserve bank of St. Louis website (https://www.stlouisfed.org/research/economists/mccracken/fred-databases) and can be found under `/input`. In the first row of the dataset, an indicator for the suggested transformations from the FRED-MD working paper (McCracken & Ng, 2015) for each variable is included. This row is handled separately in each script, without distorting the time series. For an in detail description of the dataset, please refer to the thesis.

## References
Michael W. McCracken and Serena Ng. FRED-MD: A Monthly Database for Macroeconomic Research. Technical Report 2015-012, Federal Reserve Bank of St. Louis, 2015.
