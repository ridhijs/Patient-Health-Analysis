from Datathon.Utils.getData import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import hdbscan

df = getTrainingData()
#df = pd.read_csv("imputed.csv")

df.shape
DEPENDENT_VAR = getDependentVariable()
DEPENDENT_VAR in df

df.columns[:10]

#X = df.drop([DEPENDENT_VAR ,"Unnamed: 0"] , axis=1)
X = df.drop([DEPENDENT_VAR ,"encounter_id" , "patient_id"] , axis=1)
X.shape
## Dummy Code
cats = X.apply(ptypes.is_categorical_dtype)
cats = cats[cats == True]
len(cats)

X = pd.get_dummies(X , columns=list(cats.index) , drop_first=True)
#X = X.dropna()
y = df[DEPENDENT_VAR]


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
nan_columns = np.any(np.isnan(X_train), axis=0)
nan_columns = list(nan_columns[nan_columns == False].index)
X_drop_columns = X_train[nan_columns]
scores = cross_val_score(LogisticRegressionCV(), X_drop_columns, y_train, scoring='roc_auc' , cv=3)
np.mean(scores)
# 0.8155

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel


X = df.drop([DEPENDENT_VAR ,"encounter_id" , "patient_id"] , axis=1)
X.shape
## Dummy Code
cats = X.apply(ptypes.is_categorical_dtype)
cats = cats[cats == True]
len(cats)

X = pd.get_dummies(X , columns=list(cats.index) , drop_first=True)
X = X.dropna()
y = df[DEPENDENT_VAR]

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
