%load_ext autoreload
%autoreload 2
%matplotlib inline

import xgboost as xgb

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

#from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

num_mean = SimpleImputer(strategy="mean")
df.loc[:,numeric_cols] = num_mean.fit_transform(df.loc[:,numeric_cols])

len(numeric_cols)
cat_freq = SimpleImputer(strategy="most_frequent")
df.loc[:,cat_cols] = cat_freq.fit_transform(df.loc[:,cat_cols])

df.iloc[32][df.iloc[32].isna()==True]

def getAPACHEScore(row):
    cols = {
        "age" : row["age"],
        "temperature" : row["temp_apache"],
        "heart_bpm": row["heart_rate_apache"],
        "respiratory_rate": row["resprate_apache"],
        "oxygenation": row["pao2_apache"],
        "ph": row["ph_apache"],
        "sodium": row["sodium_apache"],
        "hematocrit": row["hematocrit_apache"],
        "wbc": row["wbc_apache"],
    }
    return np.sum([calculate_single_scores(v,k) for k,v in cols.items()])


df["apacheScore"] = df.apply(getAPACHEScore , axis=1)

df["apacheScore"].describe()

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

rs = RobustScaler()
pt = PowerTransformer()
df.loc[:,numeric_cols] = rs.fit_transform(df.loc[:,numeric_cols])
df.loc[:,numeric_cols] = pt.fit_transform(df.loc[:,numeric_cols])

import plotnine as p9

p9.ggplot(df , aes(x="apacheScore" , y="apache_4a_hospital_death_prob" , color="hospital_death")) + geom_jitter()


numeric_cols = numeric_cols
numdf = df[numeric_cols]
r_in_x = numdf.corr()
r_in_x = abs(r_in_x)
distance_in_x = 1 / r_in_x
for i in range(r_in_x.shape[0]):
        distance_in_x.iloc[i, i] = 10 ^ 10


cpdist = distance_in_x.copy()

cpdist = cpdist.fillna(cpdist.max().max())
#df.isna().sum()

from scipy.spatial.distance import correlation
from sklearn.cluster import FeatureAgglomeration

corrcoefmin = 0.9
fa = FeatureAgglomeration(n_clusters=None,affinity="precomputed",compute_full_tree=True, linkage="average" ,distance_threshold=1/corrcoefmin)
fa.fit(cpdist)

numdf.shape[1]
fa.n_clusters_

fadf = pd.DataFrame({"feature":numdf.columns.values , "label":fa.labels_})

selectedFeatures = fadf.groupby("label").head(1)["feature"].values

#selectedFeatures = df.columns

DEPENDENT_VARIABLE = getDependentVariable()

pcadf = df[selectedFeatures].copy()
#ndf = df[selectedFeatures].join(df["hospital_death"])

pcadf.shape

cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
#df
pcadf = pcadf.join(df[cat_cols_minus_useless])

ndf = pcadf

cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False , handle_unknown='ignore')
endcodedNdf = ohe.fit_transform(ndf.loc[:,cols_to_dummy])
ndf = ndf.drop(cols_to_dummy,axis=1).join(pd.DataFrame(endcodedNdf))
#ndf = pd.get_dummies(ndf , columns=cols_to_dummy , drop_first=True)

ndf
#ndf[cols_to_dummy] = ndf[cols_to_dummy].astype("category")

y = ndf[DEPENDENT_VARIABLE]
X = ndf.drop(DEPENDENT_VARIABLE,axis=1)
#X = ndf[["apache_4a_hospital_death_prob" , "apacheScore"]]
X.shape

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

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

from sklearn.utils import resample
Xv,yv = resample(X_train,y_train,stratify=y_train)

cross_val_score(clf , Xv,yv,cv=5,scoring="roc_auc" ,n_jobs=3)
#array([0.89459875, 0.87728864, 0.89598546, 0.88417915, 0.88715768])

clf.fit(Xv,yv)

roc_auc_score(y_test , clf.predict(X_test))

clf.feature_importances_
featureDf = pd.DataFrame({"feature" : X_train.columns , "imp" :clf.feature_importances_})
featureDf.sort_values("imp" , ascending=False)
featureDf[featureDf["feature"] == "apacheScore"]
featureDf

cols_to_dummy[8]
tdf = getUnlabledData()

preds = godMode(tdf)

def godMode(df):
    idf = tdf.copy()
    #print(set(df.columns).difference(set(adf.columns)))
    idf.loc[:,numeric_cols] = num_mean.transform(idf.loc[:,numeric_cols])
    idf.loc[:,cat_cols] = cat_freq.transform(idf.loc[:,cat_cols])
    idf["apacheScore"] = idf.apply(getAPACHEScore , axis=1)
    idf.loc[:,numeric_cols] = rs.transform(idf.loc[:,numeric_cols])
    idf.loc[:,numeric_cols] = pt.transform(idf.loc[:,numeric_cols])
    pcadf = idf[list(selectedFeatures) + list(cat_cols_minus_useless)].copy()
    #pcadf = pcadf.join(idf[cat_cols_minus_useless])
    ndf = pcadf
    endcodedNdf = ohe.transform(ndf.loc[:,cols_to_dummy])
    ndf = ndf.drop(cols_to_dummy,axis=1).join(pd.DataFrame(endcodedNdf))
    X = ndf.drop("hospital_death" , axis=1)

    preds = clf.predict_proba(X)
    return preds

preds[:,1]
results = pd.DataFrame({"encounter_id" : tdf["encounter_id"] , "hospital_death" :preds[:,1] })
results.to_csv("./submission.csv" , index=False)
