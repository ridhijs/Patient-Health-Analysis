%load_ext autoreload
%autoreload 2
%matplotlib inline

from sklearn.decomposition import PCA
from Datathon.Utils.getData import *
from Datathon.Utils.pipeFunctions import *
from sklearn.preprocessing import StandardScaler
from plotnine import *
#df = getTrainingData()

adf = getAllImputedData()
DEPENDENT_VAR = getDependentVariable()

testDf = adf[adf["isTraining"] == 0]
df = adf[adf["isTraining"] == 1]


df = df.drop("isTraining" ,axis=1)

nas = df.isna().sum()

numeric_cols = getNumericColumns(df)
cat_cols = getCategorialColumns(df)

#from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

num_mean = SimpleImputer(strategy="mean")
df.loc[:,numeric_cols] = num_mean.fit_transform(df.loc[:,numeric_cols])

cat_freq = SimpleImputer(strategy="most_frequent")
df.loc[:,cat_cols] = cat_freq.fit_transform(df.loc[:,cat_cols])

df.shape
#df = DropCorrelated(df)
df.head()

numeric_col_types = ["APACHE covariate"  , "vitals" , "labs" , "labs blood gas"]

covariate_selected = ["albumin_apache",
"bilirubin_apache",
"bun_apache",
"creatinine_apache",
"fio2_apache",
"gcs_eyes_apache",
"gcs_motor_apache",
"gcs_verbal_apache",
"glucose_apache",
"heart_rate_apache",
"hematocrit_apache",
"map_apache",
"paco2_apache",
"paco2_for_ph_apache",
"pao2_apache",
"ph_apache",
"resprate_apache",
"sodium_apache",
"temp_apache",
"urineoutput_apache",
"wbc_apache"]

diag_cols = covariate_selected
for col_type in numeric_col_types:
    diag_cols = getColumnsByType(col_type)
    pca = PCA()
    pca.fit_transform(df.loc[:,diag_cols] )
    ratios = np.cumsum(pca.explained_variance_ratio_)
    plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
    p = (ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())
    print(col_type)
    print(p)


dbsp = ["d1_diasbp_invasive_max",
"d1_diasbp_invasive_min",
"d1_diasbp_max",
"d1_diasbp_min",
"d1_diasbp_noninvasive_max",
"d1_diasbp_noninvasive_min",
"hospital_death"
]
## Apache death probs

subdf = df[dbsp]

subdf.corr()
ggplot(subdf , aes(x="d1_diasbp_invasive_max" , y="d1_diasbp_noninvasive_max" , color="hospital_death")) + geom_jitter()


while True:
    
cols = getCategorialColumns(df)
for col_type in numeric_col_types:
    diag_cols = getColumnsByType(col_type)
    pca = PCA()
    df.loc[:,diag_cols]  = pca.fit_transform(df.loc[:,diag_cols] )
    ratios = np.cumsum(pca.explained_variance_ratio_)
    plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
    p = (ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())
    print(col_type)
    print(p)

#df.loc[:,numeric_cols].shape
df.loc[:,diag_cols].shape




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
