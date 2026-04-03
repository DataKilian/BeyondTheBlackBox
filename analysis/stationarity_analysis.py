"""
This script applies the Augemented Dickey Fuller (ADF) test to all transformed features in order to detect remaining non-stationarity.
Non-stationary features post-transformation (as suggested by McCracken and Ng (2015)) are dropped from the feature set.
The statistics from the ADF-test are saved in a CSV.

The variable HOUST in the domain-informed setting yielded p-value of 0.055 in the ADF-test. Although this is slightly above the chosen 5% significance level, the result is very close to the threshold. Given this proximity and the variable's relevance in the domain-informed specification, it was kept in the dataset rather than discarded.
The non-stationary variables in the multivariate setting were already discarded in the variable_selection.py script after the fist test. Therefore, no non-stationary variables remain here.

Output:
- description_raw.csv
- description_transformed.csv


Usage in CLI (default for the transformed dataset with s = 1):
    python analysis/stationarity_analysis.py --input data_intermediate/transformed_univariate_1.csv --output results/descriptive/stationarity_univariate_1.csv
    python analysis/stationarity_analysis.py --input data_intermediate/transformed_multivariate_1.csv --output results/descriptive/stationarity_multivariate_1.csv
    python analysis/stationarity_analysis.py --input data_intermediate/transformed_multivariate_domain_1.csv --output results/descriptive/stationarity_multivariate_domain_1.csv
"""


import argparse
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test_series(series: pd.Series):
    # catch errors in case assumptions of ADF are not met or NAs present
    try:
        result = adfuller(series.dropna(), autolag='AIC')  # AIC is used to automatically determine the lag length
        return { # return a dictionary with ADF statistics
            "adf_stat": result[0],
            "p_value": result[1],
            "lags": result[2],
            "n_obs": result[3]
        }
    except Exception: # in case of an exception return a all None dictionary
        return {
            "adf_stat": None,
            "p_value": None,
            "lags": None,
            "n_obs": None
        }

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # load dataset
    df = pd.read_csv(args.input)

    # drop first row (transformation indicators) - would distort ADF statistics
    df = df.iloc[1:].reset_index(drop=True)

    # drop sasdate if present - only numerical feature are in scope for the ADF-test
    test_cols = [c for c in df.columns if c.lower() != "sasdate"]

    results = []
    non_stationary = []

    # print all stats and save in results list
    for col in test_cols:
        print(f"column: {col}")
        series = df[col]
        result = adf_test_series(series)

        is_stationary = (
            result["p_value"] is not None and result["p_value"] < 0.05 
        )

        print(f" adf statistic: {result['adf_stat']}")
        print(f" p-value: {result['p_value']}")
        print(f" lags used: {result['lags']}")
        print(f" observations: {result['n_obs']}")
        print(" stationary? " + ("yes" if is_stationary else "no"))
        print("------------------------------------------------------------")

        results.append({
            "variable": col,
            **result,
            "stationary": is_stationary
        })

        if not is_stationary:
            non_stationary.append(col)

    # save results to CSV for further inspection
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    # summary printout
    print("\nADF SUMMARY")
    if len(non_stationary) == 0:
        print("All variables are stationary at the 5% level.")
    else: # print non-stationary variables in case some are still present
        print("Not all variables are stationary.")
        print("Non-stationary variables:")
        for col in non_stationary:
            print(f" - {col}")

    print(f"\nADF statistics saved to: {args.output}")

    # printing the output location
    print("Output written to:", args.output)
    #os.startfile(args.output) # open the csv file with saved ADF statistics

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    main()
