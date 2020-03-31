%load_ext autoreload
%autoreload 2
%matplotlib inline

import xgboost as xgb

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

savedf = df.copy()
df = savedf



cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id"]]
#df = pd.get_dummies(df , columns= cat_cols_minus , drop_first=True )


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

fa = FeatureAgglomeration(n_clusters=None,affinity="precomputed",compute_full_tree=True, linkage="average" ,distance_threshold=1/0.97)
fa.fit(cpdist)

numdf.shape[1]
fa.n_clusters_

fadf = pd.DataFrame({"feature":numdf.columns.values , "label":fa.labels_})

selectedFeatures = fadf.groupby("label").head(1)["feature"].values

#fadf = fadf.groupby("label")



DEPENDENT_VARIABLE = getDependentVariable()

#ndf = df[selectedFeatures].join(df[cat_cols_minus_useless])

pcadf = df[selectedFeatures].copy()
#ndf = df[selectedFeatures].join(df["hospital_death"])

pcadf.shape

#from sklearn.decomposition import IncrementalPCA
#fadecomp = IncrementalPCA(n_components=75)
#pcadf.iloc[:,:75] = fadecomp.fit_transform(pcadf)

pcadf = pcadf.join(df[cat_cols_minus_useless])

ndf = pcadf


#ratios = np.cumsum(fadecomp.explained_variance_ratio_)
#plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
#(ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())


#from sklearn.preprocessing import LabelEncoder


cat_cols_now = getCategorialColumns(ndf)
cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]
ndf = pd.get_dummies(ndf , columns=cols_to_dummy , drop_first=True)

#encoders = [LabelEncoder() for c in cols_to_dummy]

#for i,x in enumerate(cols_to_dummy):
#    ndf[x] = encoders[i].fit_transform(ndf[x])


y = ndf[DEPENDENT_VARIABLE]
X = ndf.drop(DEPENDENT_VARIABLE,axis=1)


data_dmatrix = xgb.DMatrix(data=X,label=y)


from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier

params = {
  'colsample_bynode': 0.78,
  'max_depth': 3,
  'num_parallel_tree': 120,
  'objective': 'binary:logistic',
  'subsample': 0.8,
  "scale_pos_weight":11,
  "base_score":0.1,
}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=2,
                    num_boost_round=10,early_stopping_rounds=5,metrics="auc", as_pandas=True, seed=123)

cv_results

model = xgb.XGBClassifier(nthreads=-1,**params)
cross_val_score(model , X,y,cv=10)
