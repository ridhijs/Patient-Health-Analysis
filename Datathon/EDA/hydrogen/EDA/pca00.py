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

from sklearn.preprocessing import LabelEncoder
cat_cols = getCategorialColumns(df)
cat_cols_dummy = [c for c in cat_cols if c not in ["encounter_id" , "patient_id"  ,"clusterId", "hospital_death"]]
#ndftotal = ndfpcadf.join(df.loc[:,cat_cols])

df = df.drop(["encounter_id" , "patient_id"  ,"clusterId"], axis=1)

numeric_cols = getNumericColumns(df)

df.loc[:,numeric_cols] = StandardScaler().fit_transform(df.loc[:,numeric_cols])

df.loc[:,cat_cols_dummy] =  df.loc[:,cat_cols_dummy].astype("str").apply(LabelEncoder().fit_transform)



df.shape

cols = df.isna().sum().sort_values(ascending=False)
cols = cols[cols > 0]
rcols = cols[cols > df.shape[0]*0.1]
mvcols = cols[~cols.isin(rcols)].index
#ndf = df.loc[:,numeric_cols]


#ndf = df.dropna(axis=1)
ndf = ReplaceColumnsWithIsMissing(df , rcols.index)
ndf.loc[:,mvcols]= ndf.loc[:,mvcols].fillna(ndf.loc[:,mvcols].mean(skipna=True))
#ndf = ndf.fillna(ndf.mean())
pcaModel = PCA().fit(ndf)


import numpy as np
import pandas as pd
from plotnine import *

ratios = np.cumsum(pcaModel.explained_variance_ratio_)
plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
(ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())


pcaModelF = PCA(n_components = 2).fit(ndf)
ndfpca = pcaModelF.transform(ndf)
ndfpca.shape

ndfpcadf = pd.DataFrame(ndfpca , columns=[f"pc{i}" for i in range(1,ndfpca.shape[1]+1)])


#ndftotal = pd.get_dummies(ndftotal , columns=cat_cols_dummy , drop_first = True , dummy_na=True)
ndftotal = ndfpcadf
ndftotal.shape

DEPENDENT_VARIABLE = getDependentVariable()




ndftotal = ndf
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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)

from math import sqrt
def r(s):
    return int(sqrt(s))

size = X_train.shape[1]

param_grid = {
        "activation" : ["logistic", "tanh", "relu"]
    , "hidden_layer_sizes" : [(12,4) , (24,8,4) , (12,) , (24)]
    ,"learning_rate":["adaptive" , "invscaling"]
}

search = RandomizedSearchCV(MLPClassifier( alpha=1e-5, random_state=1),cv=2, param_distributions=param_grid, scoring = 'roc_auc' , n_jobs=3)

search.fit(X_train , y_train)

search.best_params_
search.score(X_test , y_test)


from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
        n_jobs=-1
        ,estimators=[
        ('nn' , MLPClassifier(learning_rate="invscaling" , hidden_layer_sizes=(12,) , activation="relu"))
         ,('rf' , RandomForestClassifier(n_estimators=200))
        , ('abc' , AdaBoostClassifier(n_estimators=200)) ])

clf.fit(X_train , y_train)
roc_auc_score(y_test , clf.predict_proba(X_test)[:,1])
