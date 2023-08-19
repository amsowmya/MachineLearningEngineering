import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from data_processing import split_data

def correlation_among_numeric_feature(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()
    # get highly correlated feature and also tell to which feature it is highly correlated
    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j] > 0.8):
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features

def lr_model(X_train, y_train):
    # create a fitted model
    X_train_with_intercept = sm.add_constant(X_train)
    lr = sm.OLS(y_train, X_train_with_intercept).fit()
    return lr

def identify_significant_variables(lr, p_value_threshold=0.05):
    # print the p-values
    print(lr.pvalues)
    # print the r-squared for the model
    print(lr.rsquared)
    # print the adjusted r-squared for the model
    print(lr.rsquared_adj)
    # identify the significant variables
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars

if __name__ == "__main__":
    capped_data = pd.read_csv('ols-regression-challenge-data\capped_data.csv')
    print(capped_data.shape)

    # remove highly correlated features
    corr_features = correlation_among_numeric_feature(capped_data, capped_data.columns)
    print(corr_features)

    highly_corr_cols = [
        'state_ District of Columbia', 
        'pctpubliccoveragealone', 
        'pctempprivcoverage', 
        'upper_bound', 
        'medianagemale', 
        'lower_bound', 
        'median', 
        'popest2015', 
        'pctprivatecoveragealone', 
        'medianagefemale', 
        'pctmarriedhouseholds'
    ]

    cols = [col for col in capped_data.columns if col not in highly_corr_cols]
    len(cols)

    X_train, X_test, y_train, y_test = split_data(capped_data[cols], "target_deathrate")
    lr = lr_model(X_train, y_train)
    summary = lr.summary()
    print(summary)

    significant_vars = identify_significant_variables(lr)
    print(len(significant_vars))

    # train the model with significant variable
    significant_vars.remove("const")
    lr_sig = lr_model(X_train[significant_vars], y_train)
    summary = lr_sig.summary()
    print("==========================")
    print(summary)