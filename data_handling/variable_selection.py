"""
This script performs data loading and variable selection. 
Three different selection procedures are differentiated:
    1. univariate (only the target variable is left in the output CSV)
    2. multivariate domain (15 variables are selected based on being an economically plausible predictor of INDPRO)
    3. multivariate (a filter-based variable selection procedure aimed at discarding features with high missingness, low variance or high correlation to existing features in the feature set)

The script can be executed using three different commands, each corresponding to a specific variable selection procedure. The selected predictor variables, the target variable and the date variable are stored in a CSV file.

Output (depending on call):
- screened_univariate.csv
- screened_multivariate.csv
- screened_multivariate_domain.csv

Usage in CLI:
    python data_handling/variable_selection.py --input input/2025-12-MD.csv --output data_intermediate/screened_univariate.csv --target INDPRO --type univariate
    python data_handling/variable_selection.py --input input/2025-12-MD.csv --output data_intermediate/screened_multivariate.csv --target INDPRO --type multivariate
    python data_handling/variable_selection.py --input input/2025-12-MD.csv --output data_intermediate/screened_multivariate_domain.csv --target INDPRO --type multivariate_domain
"""

import argparse
import pandas as pd
import numpy as np
import os

# data loading
def load_data(path: str) -> pd.DataFrame:
    # load csv file
    return pd.read_csv(path, header=0)

# the following three functions define the screening steps for the multivariate variable selection procedure
# this function removes variables with low variance (below the threshold)
def remove_low_variance(df: pd.DataFrame, target, threshold: float = 1e-6) -> pd.DataFrame:
    # only include numeric variables and not the target variable
    predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != target] # number is the numerical base class in numpy
    predictors = df[predictors]
    variances = predictors.var(axis=0)
    drop = variances[variances <= threshold].index.tolist()
    if drop:
        print("removed due to low variance:")
        for v in drop:
            print(f"  - {v}")
    keep = [c for c in df.columns if c not in drop]
    return df[keep]

# this function removes variables with too many missing values (above 20%)
def remove_too_many_nas(df: pd.DataFrame, target, max_na_frac: float = 0.2) -> pd.DataFrame:
    # only include numeric variables and not the target variable
    predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != target] # number is the numerical base class in numpy
    predictors = df[predictors]
    na_frac = predictors.isna().mean()
    drop = na_frac[na_frac > max_na_frac].index.tolist()
    if drop:
        print("removed due to high missingness:")
        for v in drop:
            print(f"  - {v}")
    keep = [c for c in df.columns if c not in drop]
    return df[keep]


# this function discardes variables with a correlation > 0.95 to already included variables in the feature set
def drop_high_corr(df: pd.DataFrame, target: str, corr_threshold: float = 0.95) -> pd.DataFrame:
    predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if len(predictors) < 2:
        return df
    corr = df[predictors].corr().abs()
    to_drop = set()
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            if corr.iloc[i, j] > corr_threshold:
                col_j = predictors[j]
                to_drop.add(col_j)
    if to_drop:
        print(f"removed due to high correlation (>|{corr_threshold}|):")
        for v in to_drop:
            print(f"  - {v}")
    keep = [c for c in df.columns if c not in to_drop]
    return df[keep]

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--max_na", type=float, default=0.2)
    parser.add_argument("--var_thresh", type=float, default=1e-6)
    parser.add_argument("--corr_thresh", type=float, default=0.95)
    args = parser.parse_args()

    # load full CSV
    df = load_data(args.input)


    # separate transform row (second row)
    transforms = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

    # ensure target exists
    if args.target not in df.columns:
        raise ValueError(f"target variable '{args.target}' not found in dataset.")
    if "sasdate" not in df.columns:
        raise ValueError("Column 'sasdate' must exist in the dataset.")

    target_series = df[args.target]

    if args.type == "univariate":
        df = df[["sasdate", args.target]] # keep just date and target

    
    if args.type == "multivariate":
        # it has been detected that among the selected variables based on the defined procedure, the variable UMCSENTx is only reported quarterly from 1959 to 1977 - as the University of Michigan Costumer Sentiment is not expected to be a influential predictor, it is excluded as it would cause the series to be sparse
        df = df.drop("UMCSENTx", axis = 1)

        # screening steps
        df = remove_too_many_nas(df, args.target, args.max_na)
        df = remove_low_variance(df, args.target, args.var_thresh)
        # ensure target remains - redundant
        if args.target not in df.columns:
            df[args.target] = target_series

        df = drop_high_corr(df, args.target, args.corr_thresh)

        # reorder columns
        cols = ["sasdate", args.target] + [c for c in df.columns if c not in ["sasdate", args.target]]
        df = df[cols]

        # after performing the stationarity analysis, even after applying the suggested transformations, three variables remained non-stationary
        # those are dropped in this step to keep the stationarity script solely descriptive
        df = df.drop(["HOUSTNE", "HOUSTMW", "PERMITMW"], axis = 1)

    if args.type == "multivariate_domain":
        # chosen domain-specific variables for predicting INDPRO (as specified in the thesis)
        domain_predictors = {
        "date": ["sasdate"],
        "target": [args.target],
        "Industrial_Activity": [
            #"IPFPNSS",    # IP: Final products & nonindustrial supplies
            #"IPFINAL",    # IP: Final products
            #"IPCONGD",    # IP: Consumer Goods
            #"IPBUSEQ",    # IP: Business Equipment
            #"IPMAT",      # IP: Materials
            "IPMANSICS",  # IP Manufacturing (SIC)
            #"NAPMPI",     # ISM Manufacturing: Production Index - only quarterly
            "CUMFNS"      # Capacity Utilization: Manufacturing
        ],
        "Labor_Market": [
            #"PAYEMS",     # All employees: Total nonfarm payrolls
            #"MANEMP",     # Manufacturing employment
            #"CES0600000007", # Average weekly hours: Goods-Producing
            #"AWOTMAN",     # Avg Weekly Overtime Hours : Manufacturing
            #"AWHMAN",      # Avg Weekly Hours : Manufacturing
            #"NAPMEI",     # ISM Manufacturing: Employment Index
            #"CES0600000008", # Avg Hourly Earnings : Goods-Producing
            #"CES2000000008", # Avg Hourly Earnings : Construction
            #"CES3000000008", # Avg Hourly Earnings : Manufacturing
            "UNRATE",      # Civilian Unemployment Rate
            "CLAIMSx",      # Initial claims
            "CES0600000007", # Avg Weekly Hours : Goods-Producing
        ],
        "Consumption_Orders": [
            #"RPI",        # Real Personal Income
            #"W875RX1",    # Real personal income ex transfer receipts
            #"DPCERA3M086SBEA", # Real personal consumption expenditures
            #"CMRMTSPLx",       # Real Manu. and Trade Industries Sales
            #"NAPM",            # ISM : PMI Composite Index
            #"NAPMNOI",         # ISM: New Orders Index
            #"NAPMSDI",         # ISM: Supplier Deliveries Index
            #"NAPMII",          # ISM: Inventories Index
            #"BUSINVx",         # Total Business Inventories
            #"ISRATIOx",        # Total Business: Inventories to Sales Ratio
            "PERMIT",   # New Private Housing Permits (SAAR)
            "ANDENOx",   # New Orders for Nondefense Capital Goods
            "CMRMTSPLx"  # Real Manu. and Trade Industries Sales
        ],
        "Financial": [
            #"GS10",       # 10-year Treasury Constant Maturity Rate
            #"BAAFFM",     # Moody's BAA Corporate Bond Yield Spread
            #"AAA",        # AAA Corporate Bond Yield
            #"TB3MS",      # 3-month Treasury Bill Rate
            #"M1SL",       # M1 Money Stock
            #"FEDFUNDS",   # Effective Federal Funds Rate
            "BUSLOANS",     # Commercial and Industrial Loans
            "M2REAL",        # Real M2 Money Stock
            "COMPAPFFx",     # 3-Month Commercial Paper Minus FEDFUNDS
            "T10YFFM"       #  10-Year Treasury C Minus FEDFUNDS

        ],
        "Housing_Consumer_Sentiment": [
            "HOUST"    # Housing Starts
        ],
        "Prices": [
            #"CPIAUCSL",   # Consumer Price Index
            #"WPSFD49207",      # Producer Price Index: Finished Goods
            #"WPSID61",      # PPI: Intermediate Materials
            #"WPSID62",      # PPI: Crude Materials
            "OILPRICEx",    # Crude Oil, spliced WTI and Cushing
            #"PPICMM",       # PPI: Metals and metal products:
            #"PPIITM",      # PPI: Intermediate Materials
        ],
        "Stock market": [
            "S&P 500",   # S&Pís Common Stock Price Index
            #"S&P div yield",      # S&Pís Composite Common Stock: Dividend Yield
            #"S&P PE ratio",      # S&Pís Composite Common Stock: Price-Earnings Ratio
            #"VIXCLSx",      # VIX
        ]}

        df = df[[t for c in domain_predictors.values() for t in c]]


    # align transform row with new column order
    transform_row = pd.DataFrame([transforms.loc[df.columns]], columns=df.columns)

    # concatenate the header with the transformation indicator row and the values
    df_final = pd.concat([transform_row, df], ignore_index=True)

    # save
    df_final.to_csv(args.output, index=False)
    print(f"\nfinished. saved screened dataset to {args.output}")

    # printing the output location
    print("Output written to:", args.output)
    # # start the file
    # os.startfile(args.output)

# call the main function when the python file is run from the command line (run directly)
if __name__ == "__main__":
    main()
