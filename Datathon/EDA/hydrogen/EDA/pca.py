%load_ext autoreload
%autoreload 2
%matplotlib inline

from sklearn.decomposition import PCA
from Datathon.Utils.getData import *
from Datathon.Utils.pipeFunctions import *
from sklearn.preprocessing import StandardScaler

#df = getTrainingData()

adf = getAllImputedData()
DEPENDENT_VAR = getDependentVariable()

testDf = adf[adf["isTraining"] == 0]
df = adf[adf["isTraining"] == 1]


df = df.drop("isTraining" ,axis=1)

nas = df.isna().sum()

df.shape
#df = DropCorrelated(df)
df.shape


numeric_cols = getNumericColumns(df)

df.loc[:,numeric_cols] = StandardScaler().fit_transform(df.loc[:,numeric_cols])

df.shape

ndf = df.loc[:,numeric_cols]
ndf = ndf.fillna(ndf.mean())
pcaModel = PCA().fit(ndf)


import numpy as np
import pandas as pd
from plotnine import *

ratios = np.cumsum(pcaModel.explained_variance_ratio_)
plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
(ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())


pcaModelF = PCA(n_components = 100).fit(ndf)
ndfpca = pcaModelF.transform(ndf)
ndfpca.shape

ndfpcadf = pd.DataFrame(ndfpca , columns=[f"pc{i}" for i in range(1,ndfpca.shape[1]+1)])

from sklearn.preprocessing import LabelEncoder
cat_cols = getCategorialColumns(df)
cat_cols_dummy = [c for c in cat_cols if c not in ["encounter_id" , "patient_id"  ,"cluster_id", "hospital_death"]]
ndftotal = ndfpcadf.join(df.loc[:,cat_cols])

ndftotal["ethnicity"]
ndftotal.loc[:,cat_cols_dummy] =  ndftotal.loc[:,cat_cols_dummy].astype("str").apply(LabelEncoder().fit_transform)
#ndftotal = pd.get_dummies(ndftotal , columns=cat_cols_dummy , drop_first = True , dummy_na=True)
#ndftotal = ndfpcadf
ndftotal.shape

DEPENDENT_VARIABLE = getDependentVariable()

ndftotal[DEPENDENT_VARIABLE] = df[DEPENDENT_VARIABLE]


y = ndftotal[DEPENDENT_VARIABLE]
X = ndftotal.drop(DEPENDENT_VARIABLE,axis=1)

from Datathon.Utils.featureSelection import adaboost

adaboost(X,y,n_estimators = 50)


from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)

basetree = DecisionTreeClassifier( criterion="gini" , min_samples_split=0.4)
clf1 = AdaBoostClassifier(n_estimators=50 , learning_rate=0.5)



# 1
# basetree= DecisionTreeClassifier( criterion="entropy" , min_samples_split=0.4)
# AdaBoostClassifier(basetree,n_estimators=50 , learning_rate=0.5)
# 0.86


clf2 = GradientBoostingClassifier(init=basetree ,subsample=0.8 , max_features=0.8)


clf1.fit(X_train , y_train)
roc_auc_score(y_test , clf1.predict_proba(X_test)[:,1])


from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
        n_jobs=-1
        ,estimators=[ ('gbc' , clf2)
        , ('abc' , clf1) ])

clf.fit(X_train , y_train)
roc_auc_score(y_test , clf.predict_proba(X_test)[:,1])
