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

savedf = df.copy()
df = savedf



cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id" , "icu_id"]]
cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id"]]
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
"hospital_death" in ndf

pcadf = df[selectedFeatures].copy()
#ndf = df[selectedFeatures].join(df["hospital_death"])

pcadf.shape

#from sklearn.decomposition import IncrementalPCA
#fadecomp = IncrementalPCA(n_components=75)
#pcadf.iloc[:,:75] = fadecomp.fit_transform(pcadf)

pcadf = pcadf.join(df[cat_cols_minus_useless])

ndf = pcadf

#
#ratios = np.cumsum(fadecomp.explained_variance_ratio_)
#plotdf = pd.DataFrame({"ratio" : ratios, "nc" : np.arange(len(ratios))})
#(ggplot(plotdf , aes(x = "nc" , y="ratio")) + geom_line())


from sklearn.preprocessing import LabelEncoder


cat_cols_now = getCategorialColumns(ndf)
cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]
#ndf = pd.get_dummies(ndf , columns=cols_to_dummy , drop_first=True)

encoders = [LabelEncoder() for c in cols_to_dummy]

for i,x in enumerate(cols_to_dummy):
    ndf[x] = encoders[i].fit_transform(ndf[x])


y = ndf[DEPENDENT_VARIABLE]
X = ndf.drop(DEPENDENT_VARIABLE,axis=1)

X.shape
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)



param_grid = [
{
"n_estimators" : [100,120,140],
    "max_features":[0.2,0.3,0.4,0.5] ,
#"base_estimator__criterion":["entropy" , "gini"] ,
"max_depth":[5,7,8,9] ,
"min_samples_split":[0.4, 0.8, 0.9],
"subsample":np.arange()
#"n_estimators":[20,50,70 , 170]
}
]


best_clf = GradientBoostingClassifier( n_estimators=100 ,max_depth=4)

search = RandomizedSearchCV(best_clf ,param_grid,cv=5 , scoring="roc_auc" , n_jobs=4)
search.fit(X,y)

search.best_params_

best_clf_as = GradientBoostingClassifier( n_estimators=120 ,max_depth=8,min_samples_split=0.1088,subsample=0.78 , max_features=0.8)
cross_val_score(best_clf_as , X,y,scoring="roc_auc" ,n_jobs=3)
#array([0.89104269, 0.89996534, 0.88886178, 0.87863573, 0.88550235])
#array([0.89164067, 0.89983414, 0.89025117, 0.87912538, 0.88611378])

# 0.94 array([0.89176901, 0.90007947, 0.89029802, 0.88039547, 0.88676939])
# 0.96 + cat col dummies array([0.89441512, 0.90235087, 0.89231596, 0.88411891, 0.88896216])
# best_clf_as = GradientBoostingClassifier( n_estimators=120 ,max_depth=8,min_samples_split=0.4 , max_features=0.3)


# array([0.89329884, 0.90260717, 0.89339198, 0.88398864, 0.88807526])

#array([0.89559272, 0.90293935, 0.89513627, 0.88671052, 0.88932063])  min_samples_split=0.2
# array([0.89742626, 0.90363917, 0.89581721, 0.88748908, 0.89007127]) 0.15
 # array([0.89699183, 0.90387244, 0.89587974, 0.88873841, 0.89132885]) 0.12
 # array([0.89635552, 0.90418434, 0.89624599, 0.88819072, 0.89015114]) 0.11

 # subsample 0.5 array([0.89354124, 0.90158949, 0.89308373, 0.88547431, 0.88900445])
 # subsample 0.9 array([0.89740998, 0.90304134, 0.89559528, 0.88801454, 0.89145731])
