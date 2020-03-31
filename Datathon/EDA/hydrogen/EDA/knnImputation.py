%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import hdbscan

#df = getTrainingData()
df = pd.read_csv("imputed.csv")
DEPENDENT_VAR = getDependentVariable()
DEPENDENT_VAR in df

#bogus_cols =  ["encounter_id" , "patient_id"]
bogus_cols =  ["Unnamed: 0"]
numeric_cols = getNumericColumns(df)
cat_cols = getCategorialColumns(df)
cat_cols = list(filter(lambda x : x not in bogus_cols + ["hospital_death"] , cat_cols))
#corr = df.corr()

#df.loc[:,numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
df = pd.get_dummies(df , columns=cat_cols ,dummy_na=True)
df =df.drop(bogus_cols , axis=1)


kdf = df.copy()
#kdf = df.sample(50000)
kdf.shape
kdf = kdf.astype("float")
kdf = kdf.fillna(0)
#kdf.head()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10, copy=False)
imputer.fit_transform(kdf )

kdf.shape
impdf = pd.DataFrame(kdf , columns=kdf.columns)
#kdf.loc[:,numeric_cols] = kdf[numeric_cols].fillna(0 )
#from sklearn.metrics.pairwise import euclidean_distances
#dist = euclidean_distances(kdf, kdf)
#dist.shape
DEPENDENT_VAR in impdf.columns

y = impdf[DEPENDENT_VAR]
X = impdf.drop("hospital_death" , axis=1)

y.value_counts()


## Feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier


def run(X,y,weights={0:1 , 1:9} , n_estimators = 600):
    X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
    rfc = RandomForestClassifier(criterion="entropy" , class_weight=weights, random_state=60 , n_estimators=n_estimators)
    rfc.fit(X_train , y_train)

    model = SelectFromModel(rfc, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)
    print(roc_auc_score(y_test , rfc.predict_proba(X_test)[:,1]))
    return X_new


def adaboost(X,y , n_estimators=100):
    X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
    clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X, y)
    print(roc_auc_score(y_test , clf.predict_proba(X_test)[:,1]))
    return clf

X_n = run(X,y)

X_n2 = run(X_n , y)

X_n3 = run(X_n2 , y)

X_n4 = run(X_n3 , y , {0:1 , 1:5} , 300)

X_n4

adaboost(X,y,200)
adaboost(X_n2 , y , 200)
