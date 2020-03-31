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

#df.loc[:,cat_cols_dummy] =  df.loc[:,cat_cols_dummy].astype("str").apply(LabelEncoder().fit_transform)

df = pd.get_dummies(df , columns=cat_cols_dummy , drop_first = True , dummy_na=True)


df.shape

cols = df.isna().sum().sort_values(ascending=False)
cols = cols[cols > 0]
rcols = cols[cols > df.shape[0]*0.1]
mvcols = cols[~cols.isin(rcols)].index
#ndf = df.loc[:,numeric_cols]


#ndf = df.dropna(axis=1)
ndf = ReplaceColumnsWithIsMissing(df , rcols.index)
ndf.loc[:,mvcols]= ndf.loc[:,mvcols].fillna(ndf.loc[:,mvcols].mean(skipna=True))

DEPENDENT_VARIABLE = getDependentVariable()

ndftotal = ndf
ndftotal[DEPENDENT_VARIABLE] = df[DEPENDENT_VARIABLE]



y = ndftotal[DEPENDENT_VARIABLE]
X = ndftotal.drop(DEPENDENT_VARIABLE,axis=1)



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


param_grid = {
        "activation" : ["relu" ,"logistic"]
    , "hidden_layer_sizes" : [(48,) ,(24,) ,(12,) , (1,), (1 ,2),(1 ,2 , 2) , (1 ,2,3), (1 ,2 , 4)]
    ,"learning_rate":[ "constant","invscaling" , "adaptive"]
    , "alpha" : [ 1e-4 , 1e-5 , 1e-6 , 1e-7]
}


baseMLP = MLPClassifier(random_state=1 , max_iter=1000)
search = RandomizedSearchCV(baseMLP,cv=2, param_distributions=param_grid, scoring = 'roc_auc' , n_jobs=4)

search.fit(X_train , y_train)

search.best_params_
roc_auc_score(y_test , search.predict_proba(X_test)[:,1])
