from Datathon.Utils.getData import *
from Datathon.Utils.getPreparedData import *

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import hdbscan

df = getTrainingData()
DEPENDENT_VAR = getDependentVariable()
DEPENDENT_VAR in df

numericCols = getNumericColumns()
colsWithLargeMissingValues = df.loc[:,df.isna().sum() > df.shape[0] * 0.6]


# clusterer = hdbscan.HDBSCAN(min_cluster_size=1000)
# #kdf["cluster"]=  clusterer.fit_predict(kdf.iloc[:,:10].isnull())
# df["cluster"]=  clusterer.fit_predict(colsWithLargeMissingValues.isnull())
# df["cluster"] = df["cluster"].astype("category")
#
# #df.to_csv("training_with_cluster.csv")
# dfc = pd.read_csv("training_with_cluster.csv")
# df["cluster"] = dfc["cluster"]


            #numericCols = [c for c in numericCols if c not in [maxCol , minCol]]

def getOutliersScore(df , numericCols):
    scaler = StandardScaler()
    #catDf = df.drop(numericCols , axis=1)
    numDf = df.loc[:,numericCols]
    numDf = numDf[(numDf > -3) & (numDf < 3)]
    df["outlierScore"] = numDf.apply(lambda x: x.isna().sum(), axis=1)
    df.loc[:, "outlierScore"] = list(scaler.fit_transform( df[["outlierScore"]]))
    return df
    #return numDf.join(catDf)

def impute(series):
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
#
# def imputeByCluster(series):
#     if ptypes.is_numeric_dtype(series):
#         series.fillna( series.mean(skipna=True) , inplace=True)
#     if ptypes.is_categorical_dtype(series):
#         imputeVal = df[df[DEPENDENT_VAR] == 1][series.name].value_counts().sort_values().index[0]
#         series.fillna(imputeVal , inplace=True)
#
#     return series
#
# def nimpute(fdf):
#     return fdf.apply(imputeByCluster,axis=0)

#df = df.groupby("cluster").apply(nimpute)

df = df.apply(impute ,axis=0)

df = diffCols(df , numericCols)
df.shape

df[DEPENDENT_VAR].isna().sum()

df.isna().sum()[df.isna().sum() > 0].index

# means = df[df[DEPENDENT_VAR] == 1].mean(skipna =True)
# df[numericCols] = df[numericCols].fillna( means)

from sklearn.preprocessing import PowerTransformer
df.loc[:,numericCols] = PowerTransformer().fit_transform(df.loc[:,numericCols])

#df[((df[numericCols] > 3) | (df[numericCols] < -3)).any(axis=1)] = df[((df[numericCols] > 3) & (df[numericCols] < -3)).all(axis=1)]


df = getOutliersScore(df, numericCols)
df["outlierScore"]
df.shape


from plotnine import *

(ggplot(df)
    + aes(x = 'weight')
    + geom_histogram())


from sklearn.model_selection import train_test_split

# colsToInclude = ['apache_4a_icu_death_prob',
#  'apache_4a_hospital_death_prob',
#  'd1_lactate_min',
#  'd1_spo2_min',
#  'd1_sysbp_min',
#  'd1_lactate_max',
#  'd1_arterial_ph_min',
#  'gcs_motor_apache',
#  'd1_sysbp_noninvasive_min',
#  'temp_apache',
#  'd1_heartrate_min',
#  'apache_2_diagnosis_114.0',
#  'd1_temp_min',
#  'd1_bun_max',
#  'd1_mbp_noninvasive_min',
#  'gcs_eyes_apache',
#  'd1_bun_min',
#  'd1_temp_max',
#  'd1_mbp_min',
#  'bun_apache',
#  'd1_creatinine_max',
#  'creatinine_apache',
#  'd1_platelets_min',
#  'ph_apache',
#  'd1_heartrate_max',
#  'bmi',
#  'd1_hco3_min',
#  'd1_arterial_ph_max',
#  'heart_rate_apache',
#  'age',
#  'd1_pao2fio2ratio_min',
#  'd1_wbc_min',
#  'd1_platelets_max',
#  'd1_pao2fio2ratio_max',
#  'd1_resprate_min',
#  'd1_creatinine_min',
#  'd1_glucose_min',
#  'weight',
#  'd1_hco3_max',
#  'pre_icu_los_days',
#  'd1_arterial_pco2_min',
#  'd1_diasbp_noninvasive_min',
#  'd1_wbc_max',
#  'gcs_verbal_apache',
#  'd1_diasbp_min',
#  'h1_temp_min',
#  'd1_sysbp_max',
#  'h1_resprate_min',
#  'd1_arterial_po2_min',
#  'fio2_apache',
#  'glucose_apache',
#  'map_apache',
#  'h1_temp_max',
#  'wbc_apache',
#  'd1_sysbp_noninvasive_max',
#  'h1_sysbp_min',
#  'd1_resprate_max',
#  'd1_sodium_max',
#  'd1_glucose_max',
#  'd1_albumin_min']
colsToInclude = df.columns
#colsToExclude = [DEPENDENT_VAR , "encounter_id" , "patient_id"  , "icu_id" , "apache_3j_diagnosis"]
colsToExclude = [DEPENDENT_VAR , "encounter_id" , "patient_id"]
colsToExclude += [ c for c in df.columns if c not in colsToInclude]
#df["intercept"] = 1
Y = df[DEPENDENT_VAR]
X = df.drop(columns = colsToExclude)

X.shape
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder(drop="first")
cats = X.apply(ptypes.is_categorical_dtype)
cats = cats[cats == True]

X = pd.get_dummies(X , columns=list(cats.index) , drop_first=True)
X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.33, stratify=Y)

y_train.value_counts(normalize=True)
from sklearn.metrics import accuracy_score , confusion_matrix ,roc_curve,roc_auc_score ,precision_score, recall_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression


logit = LogisticRegression(max_iter = 10000)
logit.fit(X_train , y_train)



preds = logit.predict(X_test)
probs_y=logit.predict_proba(X_test)[:,1]

# for thresh in np.linspace(0, 1, num=30):
#     preds = np.where(probs_y > thresh,1, 0)
#     #y_test_dum = np.where(y_test == 1,1, 0)
#     #print(preds, y_test_dum)
#     a=accuracy_score(y_test , preds)
#     r=roc_auc_score(y_test , preds)
#     rec=recall_score(y_test , preds)
#     p = precision_score(y_test,preds)
#     print(f"thresh : {thresh} acc : {a} auc: {r} recall : {rec} precision: {p}")



fpr, tpr, thresholds = metrics.roc_curve(y_test , probs_y ,drop_intermediate=False, pos_label=1)
rocdf = pd.DataFrame({"fpr" : fpr , "tpr" : tpr , "thresholds" : thresholds})

(ggplot(rocdf , aes(x='fpr' , y='tpr', fill='thresholds')) + geom_line())

accuracy_score(y_test , preds)
roc_auc_score(y_test , probs_y)

rocdf[rocdf["fpr"] < 0.25].sort_values("tpr" , ascending=False)
import statsmodels.api as sm
logitstat = sm.Logit(y_train.astype(float), X_train.astype(float))
logitstat.fit()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train , y_train)
roc_auc_score(y_test , rf.predict_proba(X_test)[:,1])


importances = pd.DataFrame({"imp"  : rf.feature_importances_ , "feature" : X_train.columns})
importances.sort_values("imp" , ascending=False).head(60)

list(importances.sort_values("imp" , ascending=False).head(60)["feature"])

importances[importances["feature"] == "cluster"]


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train , y_train)
roc_auc_score(y_test , ada.predict_proba(X_test)[:,1])
