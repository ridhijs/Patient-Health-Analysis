%load_ext autoreload
%autoreload 2
%matplotlib inline


from sklearn.decomposition import PCA
from Datathon.Utils.apache2Calc import *
from Datathon.Utils.getData import *
from Datathon.Utils.pipeFunctions import *
from sklearn.preprocessing import StandardScaler
from plotnine import *
#df = getTrainingData()

adf = getTrainingData()



df= adf.copy()
numeric_cols = getNumericColumns(df)
cat_cols = getCategorialColumns(df)


df["apache_3j_diagnosis"] = df["apache_3j_diagnosis"].astype(str).apply(lambda x: x.split(".")[0])

from sklearn.impute import SimpleImputer

num_mean = SimpleImputer(strategy="mean")
df.loc[:,numeric_cols] = num_mean.fit_transform(df.loc[:,numeric_cols])

len(numeric_cols)
cat_freq = SimpleImputer(strategy="most_frequent")
df.loc[:,cat_cols] = cat_freq.fit_transform(df.loc[:,cat_cols])


cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , mutual_info_classif

lbe = OrdinalEncoder()

df.loc[:,cols_to_dummy] = lbe.fit_transform(df.loc[:,cols_to_dummy])

DEPENDENT_VARIABLE = getDependentVariable()

y = df[DEPENDENT_VARIABLE]
X = df.drop([DEPENDENT_VARIABLE] + numeric_cols,axis=1)

X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)

fs = SelectKBest(score_func=mutual_info_classif, k='all')
fs.fit(X_train, y_train)

fs.scores_


from plotnine import *


minfopdf = pd.DataFrame({"cols":X_train.columns.values , "scores":fs.scores_})
minfopdf[minfopdf["scores"] > 0.01].shape

xcatfs = minfopdf[minfopdf["scores"] > 0.01]["cols"].values

ggplot(minfopdf.sort_values("scores" , ascending=False).iloc[:200] , aes(x= "cols" , y="scores" )) +geom_col() + coord_flip()


from sklearn.decomposition import PCA

kpca = PCA()
kpca.fit_transform(df[numeric_cols])

kpca.explained_variance_

pcapdf = pd.DataFrame({"cols":np.arange(0,len(numeric_cols)) , "explained":np.cumsum(kpca.explained_variance_ratio_)})

ggplot(pcapdf , aes(x="cols" , y="explained")) + geom_line()

kpca = PCA(n_components=50)
xft = kpca.fit_transform(df[numeric_cols])

xft.shape


fsfdf = pd.DataFrame(xft).join(df[xcatfs])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore' , sparse=False)

#df[xcatfs].apply(lambda x: len(x.unique()))
ohedf = ohe.fit_transform(df[xcatfs])
catfsdf = pd.DataFrame(ohedf , columns=[f"cat_{i}" for i in range(ohedf.shape[1])])

#finalfsdf = pd.DataFrame(xft).join(catfsdf)
finalfsdf = pd.DataFrame(xft)

y = df[DEPENDENT_VARIABLE]
X = finalfsdf

import lightgbm as lgb

params = {
  'max_depth': 10,
  'n_estimators ': 10,
  'objective': 'binary',
  'colsample_bytree': 0.8,
  "class_weight":{0:1 , 1:20},
  "base_score":0.2,
  "n_jobs":-1,
  "metric":"auc",
  "reg_alpha":0.4,
  "reg_lambda":0.18,
}

clf = lgb.LGBMClassifier(**params)

from sklearn.model_selection import learning_curve


train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
train_sizes

lcurveplotdf = pd.DataFrame({"train_size":train_sizes , "train_score" : train_scores[:,1] , "valid_score":valid_scores[:,1]})

ggplot(lcurveplotdf ) + \
    geom_line(aes(x="train_size" , y="train_score") , color="red") + \
    geom_line(aes(x="train_size" , y="valid_score") , color="green")
