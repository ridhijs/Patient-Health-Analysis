%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *
from Datathon.Utils.getPreparedData import *
from Datathon.Utils.featureSelection import *
from Datathon.Utils.pipeFunctions import *
from sklearn.preprocessing import PowerTransformer




adf = getAllImputedData()
DEPENDENT_VAR = getDependentVariable()

testDf = adf[adf["isTraining"] == 0]
df = adf[adf["isTraining"] == 1]


df = df.drop("isTraining" ,axis=1)

nas = df.isna().sum()

df.shape
df = DropCorrelated(df)




df.shape

nas = df.isna().sum()
extremeMissing = nas[nas > df.shape[0]*0.1]
df = ReplaceColumnsWithIsMissing(df , extremeMissing.index)
#df = df.drop(extremeMissing.index, axis=1)

bogus_cols = ["encounter_id" , "patient_id","icu_id" , "hospital_id"]
cat_cols = [c for c in getCategorialColumns(df) if c not in bogus_cols]
#cat_cols = cat_cols + ["clusterId"]

cat_cols_to_dummy = [c for c in cat_cols if c not in [ "hospital_death"]]

numeric_cols = getNumericColumns(df)
means = df.loc[:, numeric_cols].mean(skipna=True)

df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(means)
df.loc[:, numeric_cols] = PowerTransformer().fit_transform(df.loc[:, numeric_cols])

#jdf = diffCols(df , numeric_cols)
#df = diffCols(df , numeric_cols)

replaceCats = df.loc[:,cat_cols].apply(lambda x: x.value_counts().sort_values().index[0])
df.loc[:,cat_cols] = df.loc[:,cat_cols].fillna(replaceCats)
df = pd.get_dummies(df , columns =cat_cols_to_dummy , drop_first=True )

df.shape


y = df[DEPENDENT_VAR]
X = df.drop(DEPENDENT_VAR , axis=1)
X.shape

nas = X.isna().sum()
nas[nas > 0]

X_new = RF(X,y)

X_new2 = RF(X_new,y)

adaboost(X_new,y ,n_estimators=200, learning_rate=0.8)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
basetree = DecisionTreeClassifier( criterion="entropy")

#clf = GradientBoostingClassifier(basetree , subsample=0.5 , max_features=0.8).fit(X_train , y_train)

basetree.get_params()
param_grid = {"init__min_samples_split" : [ 0.3 , 0.4 , 0.5],
              #"base_estimator__max_depth" :   [1],
              #"base_estimator__class_weight" :   [{0:1 , 1:1}],
              "subsample": [0.5 , 0.6 , 0.8],
              "max_features":[0.5 , 0.6 , 0.8 , 1]
             }

#clf = AdaBoostClassifier(basetree , n_estimators=200 , learning_rate=1)

clf = GradientBoostingClassifier(init=basetree)
search = RandomizedSearchCV(clf,cv=2, param_distributions=param_grid, scoring = 'roc_auc' , n_jobs=3)

search.fit(X_train , y_train)


# {'n_estimators': 100,
#  'base_estimator__min_samples_split': 0.2,
#  'base_estimator__max_depth': 1,
#  'base_estimator__class_weight': {0: 2, 1: 1}}


## GradientBoostingClassifier
## {'subsample': 0.8, 'max_features': 0.8, 'init__min_samples_split': 0.4}

search.score(X_test, y_test)
search.best_params_


clf.fit(X_train , y_train)
clf.score(X_test , y_test)

list(X_train.columns[np.argsort(-clf.feature_importances_  , )])

roc_auc_score(y_test , clf.predict_proba(X_test)[:,1])


from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
        n_jobs=-1
        ,estimators=[ ('rf' , RandomForestClassifier(n_estimators=200))
        , ('abc' , AdaBoostClassifier(n_estimators=200)) ])

clf.fit(X_train , y_train)
roc_auc_score(y_test , clf.predict_proba(X_test)[:,1])
