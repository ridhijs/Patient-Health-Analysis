import pandas as pd
import numpy as np

import pandas.api.types as ptypes
import os
from pathlib import Path

cwd = Path().resolve()
cwd

pathToRoot =  os.path.abspath(os.path.join(cwd ,"Datathon"))
pathToDataFolder =  os.path.abspath(os.path.join(cwd ,"Datathon", "Data"))
trainDataPath = os.path.abspath(os.path.join(pathToDataFolder, "training_v2.csv"))
unlabledDataPath = os.path.abspath(os.path.join(pathToDataFolder, "unlabeled.csv"))
imputedDataPath = os.path.abspath(os.path.join(cwd , "impute_complete.csv"))
path = os.path.abspath(os.path.join(pathToDataFolder, "WIDS Datathon 2020 Dictionary.csv"))

DICTDF = pd.read_csv(path)

DICTDF[DICTDF["Variable Name"] == "apache_2_diagnosis"]

DICTDF[DICTDF["Data Type"] =="string"]

DICTDF["Category"].unique()

DICTDF["Data Type"].unique()
DICTDF.groupby("Category").head(5)

def _columnDtypeCorrection(c):
    #print(c)
    #print(c)
    if c == "isTraining":
        dtype = "binary"
    elif c == "clusterId":
        dtype = "string"
    else:
        dtype = list(DICTDF[DICTDF["Variable Name"] == c]["Data Type"])[0]

        if c.find("_id") > -1:
            dtype = "binary"
        if c.find("bmi") > -1:
            dtype = "float"

    #print(c,dtype)
    return dtype


def _dataTypeCorrection(df):
    for c in df.columns:
        dtype = _columnDtypeCorrection(c)
        if dtype == "string" or dtype == "binary":
            df[c] = df[c].astype("category")
        elif dtype == "integer" and len(df[c].unique()) < 5:
            df[c] = df[c].astype("category")
        else:
            df[c] = df[c].astype("float")
    return df

def _getData(filepath):
    df =  pd.read_csv(filepath)
    return _dataTypeCorrection(df)

def getTrainingData():
    return _getData(trainDataPath)

def getUnlabledData():
    return _getData(unlabledDataPath)

def getAllImputedData():
    adf = getAllData()
    numeric_cols = getNumericColumns(adf)
    idf = pd.read_csv(imputedDataPath)
    adf.loc[:,numeric_cols] = idf.loc[:,numeric_cols]
    adf["clusterId"] = idf["clusterId"].astype("category")
    adf = _dataTypeCorrection(adf)
    return adf

def getAllData():
    df = getTrainingData()
    udf = getUnlabledData()

    df["isTraining"] = 1
    udf["isTraining"] = 0
    udf["hospital_death"] = np.nan
    adf = pd.concat([df,udf])

    adf = _dataTypeCorrection(adf)
    adf["isTraining"] = adf["isTraining"].astype("category")
    adf = adf.reset_index(drop=True)
    return adf

def getDiagnosisColumns():
    return ["apache_2_diagnosis" , "apache_2_bodysystem" , "icu_type"]

def getNumericColumns(df):
    return [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]

def getCategorialColumns(df):
    return [c for c in df.columns if ptypes.is_categorical_dtype(df[c])]

def getColumnTypes():
    return DICTDF["Category"].unique()


def getColumnsByType(type):
    return list(DICTDF[DICTDF["Category"] == type]["Variable Name"])

def getDependentVariable():
    return "hospital_death"


#getColumnTypes()


#getColumnsByType("demographic")

#getTrainingData().columns
DICTDF[DICTDF["Variable Name"] == "bmi"]
