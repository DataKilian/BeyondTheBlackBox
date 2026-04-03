"""
This script reads the screened CSV (output of variable_selection.py) and applies the standard FRED-MD transformations as suggested by McCracken & Ng (2015). Additionally, data cleaning steps are executed to deal with missing values. Ultimately, a CSV containing the transformed variables gets saved.
1-period differences for the baseline 1-month ahead forecasting setting get computed. For further information, please refer to the thesis.

The following transformations are applied correspondingly to the suggestion by McCracken & Ng (2015):
1) no transformation
2) first difference: Δx_t
3) second difference: Δ²x_t
4) log(x_t)
5) first difference of log: Δlog(x_t)
6) second difference of log: Δ²log(x_t)
7) detrended growth transforamtion: Δ(x_t/x_{t-1}-1)

Output (depending on call):
- transformed_univariate_1.csv
- transformed_multivariate_11.csv
- transformed_multivariate_domain_1.csv

Usage in CLI:
    python data_handling/variable_transformation_1.py --input data_intermediate/screened_univariate.csv --output data_intermediate/transformed_univariate_1.csv
    python data_handling/variable_transformation_1.py --input data_intermediate/screened_multivariate.csv --output data_intermediate/transformed_multivariate_1.csv
    python data_handling/variable_transformation_1.py --input data_intermediate/screened_multivariate_domain.csv --output data_intermediate/transformed_multivariate_domain_1.csv
"""

import argparse
import pandas as pd
import numpy as np
import os

TRANSFORM_LAG = 1  # monthly transformation span for the baseline setup


# definition of the seven transforamtion functions (1-step differences)
def transform_none(x):
    return x

def transform_diff1(x):
    return x.diff(TRANSFORM_LAG)

def transform_diff2(x):
    return x.diff(TRANSFORM_LAG).diff(TRANSFORM_LAG)

def transform_log(x):
    return np.log(x)

def transform_log_diff1(x):
    return np.log(x).diff(TRANSFORM_LAG)

def transform_log_diff2(x):
    return np.log(x).diff(TRANSFORM_LAG).diff(TRANSFORM_LAG)

def transform_growth_diff(x):
    return ((x / x.shift(TRANSFORM_LAG) - 1).diff(TRANSFORM_LAG))


# mapping transformation codes
TRANSFORM_FUNCS = {
    1: transform_none,
    2: transform_diff1,
    3: transform_diff2,
    4: transform_log,
    5: transform_log_diff1,
    6: transform_log_diff2,
    7: transform_growth_diff,
}


# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "sasdate" not in df.columns:
        raise ValueError("Input CSV must contain a 'sasdate' column")

    # extract the transformation row (row 0) - indicating the suggested transformation
    transform_row = df.iloc[0].copy()
    df_data = df.iloc[1:].reset_index(drop=True)

    # apply the transformations
    transformed_columns = {"sasdate": df_data["sasdate"]}

    for col in df_data.columns:
        if col == "sasdate":
            continue

        code = int(transform_row[col])
        func = TRANSFORM_FUNCS[code]

        transformed_columns[col] = func(df_data[col])

    # create transformed dataframe out of dictionary
    transformed_df = pd.DataFrame(transformed_columns)

    # drop all NA rows at the top of the dataframe, which appear due to the differences in the applied transformation (e.g. first vs. second degree differencing- compresses 2 vs. 4 values)
    # find the index of the row that is the first not fully NA
    first_full_idx = transformed_df.notna().all(axis=1).idxmax()

    # slice the dataframe from that row onwards
    transformed_df = transformed_df.iloc[first_full_idx:].reset_index(drop=True)

    # insert transformation indicator row as first row again
    transform_row_df = pd.DataFrame([transform_row], columns=transform_row.index)

    # concat so transform row becomes the first row
    final_df = pd.concat([transform_row_df, transformed_df], ignore_index=True)

    # save CSV
    final_df.to_csv(args.output, index=False)
    print(f"Transformed dataset saved to {args.output}")

    # printing the output location
    print("Output written to:", args.output)
    # # open CSV
    # os.startfile(args.output)

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    main()