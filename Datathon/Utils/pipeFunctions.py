import pandas.api.types as ptypes
import numpy as np

from Datathon.Utils.getData import *

DEPENDENT_VAR = getDependentVariable()

def impute(df,series):
    #cat = DICTDF[DICTDF["Variable Name"] == series.name]["Category"].values[0]
    # if cat.find("lab") > -1 or cat.find("vital") > -1:
    #     imputeVal = df[df[DEPENDENT_VAR] == 1][series.name].mean()
    #     series.fillna(imputeVal , inplace=True)
    if ptypes.is_numeric_dtype(series):
        # if series.name in colsWithLargeMissingValues.columns:
        #     imputeVal = df[df[DEPENDENT_VAR] == 1][series.name].mean()
        # else:
        imputeVal = series.mean()
        series.fillna(imputeVal , inplace=True)
    if ptypes.is_categorical_dtype(series):
        #imputeVal = series.value_counts().sort_values().index[0]
        #series.fillna(imputeVal , inplace=True)
        #imputeVal = "missing"
        #series = series.astype(str).fillna("missing").astype("category")
        imputeVal = df[df[DEPENDENT_VAR] == 1][series.name].value_counts().sort_values().index[0]
        series.fillna(imputeVal , inplace=True)

    return series

def ReplaceColumnsWithIsMissing(df, cols):
    for c in cols:
        df[c+"_is_missing"] = df[c].isna()
        df[c+"_is_missing"] = df[c+"_is_missing"].astype("category")
    df = df.drop(cols,axis=1)
    return df

def DropCorrelated(df):
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


    df= df.drop(to_drop , axis=1)
    return df
