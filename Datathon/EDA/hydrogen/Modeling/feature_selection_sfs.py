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


from sklearn.impute import SimpleImputer

num_mean = SimpleImputer(strategy="mean")
df.loc[:,numeric_cols] = num_mean.fit_transform(df.loc[:,numeric_cols])

len(numeric_cols)
cat_freq = SimpleImputer(strategy="most_frequent")
df.loc[:,cat_cols] = cat_freq.fit_transform(df.loc[:,cat_cols])


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

rs = RobustScaler()
pt = PowerTransformer()
df.loc[:,numeric_cols] = rs.fit_transform(df.loc[:,numeric_cols])
df.loc[:,numeric_cols] = pt.fit_transform(df.loc[:,numeric_cols])

cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , mutual_info_classif

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore' , sparse=False)
ohedf = ohe.fit_transform(df[cols_to_dummy])

ndf = pd.DataFrame(ohedf).join(df)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

DEPENDENT_VARIABLE = getDependentVariable()
y = ndf[DEPENDENT_VARIABLE]
X = ndf.drop([DEPENDENT_VARIABLE] + cat_cols_minus_useless,axis=1)

sfs1 = SFS(knn,
           k_features=50,
           forward=True,
           floating=True,
           verbose=2,
           scoring='roc_auc',
           cv=3,n_jobs=2)

sfs1 = sfs1.fit(X, y , custom_feature_names=X.columns.values)


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
