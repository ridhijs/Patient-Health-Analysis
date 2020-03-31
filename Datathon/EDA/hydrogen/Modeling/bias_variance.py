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

ndf = df.copy()

cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
#df
#pcadf = pcadf.join(df[cat_cols_minus_useless])

#ndf = pcadf

cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False , handle_unknown='ignore')
endcodedNdf = ohe.fit_transform(ndf.loc[:,cols_to_dummy])
ndf = ndf.drop(cols_to_dummy,axis=1).join(pd.DataFrame(endcodedNdf))

ndf.shape

from sklearn.decomposition import PCA
pcam = PCA()
pcam.fit(ndf)


evr = np.cumsum(pcam.explained_variance_ratio_)
pcam.components_[:2 , :]

def FeatureSelection(df,numeric_cols , corrCoefThres=0.9):
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

    corrcoefmin = corrCoefThres
    fa = FeatureAgglomeration(n_clusters=None,affinity="precomputed",compute_full_tree=True, linkage="average" ,distance_threshold=1/corrcoefmin)
    fa.fit(cpdist)

    numdf.shape[1]
    fa.n_clusters_

    fadf = pd.DataFrame({"feature":numdf.columns.values , "label":fa.labels_})

    selectedFeatures = fadf.groupby("label").head(1)["feature"].values
    return selectedFeatures


DEPENDENT_VARIABLE = getDependentVariable()

pcadf = df[selectedFeatures].copy()





sf = FeatureSelection(ndf , ndf.columns)
len(sf)

ndf = ndf[sf]
y = ndf[DEPENDENT_VARIABLE]
X = ndf.drop(DEPENDENT_VARIABLE,axis=1)

X.columns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
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

X.shape
train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
train_sizes
train_scores
valid_scores

train_scores[:,1]
plotdf = pd.DataFrame({"train_size":train_sizes , "train_score" : train_scores[:,1] , "valid_score":valid_scores[:,1]})

ggplot(plotdf ) + \
    geom_line(aes(x="train_size" , y="train_score") , color="red") + \
    geom_line(aes(x="train_size" , y="valid_score") , color="green")
