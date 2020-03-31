from Datathon.Utils.getData import *

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import hdbscan

import time


def diffCols(df, numericCols):
    for maxCol in numericCols:
        if maxCol.find("max") > -1:
            name = "_".join(maxCol.split("_")[1:-1])
            minCol = [c for c in df.columns if c.find(name) > -1 and c.find("min") > -1][0]
            diffCol = f"{name}_diff"
            maxOutlier = f"{maxCol}_isoutlier"
            minOutlier = f"{minCol}_isoutlier"
            minMaxRatio = f"{name}_minMaxRatio"
            meanCol = f"{name}_mean"
            df[diffCol] = df[maxCol] - df[minCol]
            #df[minMaxRatio] = df[maxCol] / df[minCol]
            df[maxOutlier] = df[maxCol] > df[maxCol].mean() + df[maxCol].std()*3
            df[minOutlier] = df[minCol] < df[minCol].mean() - df[maxCol].std()*3
            df[meanCol] = (df[maxCol] + df[minCol])/2
            df[maxOutlier] = df[maxOutlier].astype("category")
            df[minOutlier] = df[minOutlier].astype("category")

            #df = df.drop([maxCol , minCol],axis=1)
            #numericCols.append(diffCol)
    return df

def getPreparedAllData():
    df = getAllData()
    DEPENDENT_VAR = getDependentVariable()
    DEPENDENT_VAR in df
    "isTraining" in df.columns

    bogus_cols =  ["encounter_id" , "patient_id"]
    numeric_cols = getNumericColumns(df)

    "isTraining" not  in numeric_cols

    cat_cols = getCategorialColumns(df)
    cat_cols = list(filter(lambda x : x not in bogus_cols + ["hospital_death" , "isTraining"] , cat_cols))
    #corr = df.corr()

    len(cat_cols + numeric_cols)
    df.loc[:,numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    df.loc[:,numeric_cols] = PowerTransformer(method="yeo-johnson").fit_transform(df[numeric_cols])

    df = pd.get_dummies(df , columns=cat_cols ,dummy_na=True , drop_first=True)
    df =df.drop(bogus_cols , axis=1)
    return df


def dummyCodeDf(df):
    bogus_cols =  ["encounter_id" , "patient_id"]
    cat_cols = getCategorialColumns(df)
    cat_cols = list(filter(lambda x : x not in bogus_cols + ["hospital_death" , "isTraining"] , cat_cols))
    return pd.get_dummies(df , columns=cat_cols ,dummy_na=True)
