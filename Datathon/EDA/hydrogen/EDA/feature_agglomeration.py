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

cat_cols_minus = [c for c in cat_cols if c not in ["hospital_death", "encounter_id" , "hospital_id" , "patient_id" , "icu_id"]]
df = pd.get_dummies(df , columns= cat_cols_minus)


df.shape


from sklearn.cluster import FeatureAgglomeration


DEPENDENT_VARIABLE = getDependentVariable()

y = df[DEPENDENT_VARIABLE]
X = df.drop(DEPENDENT_VARIABLE,axis=1)


from sklearn.tree import DecisionTreeClassifier
basetree = DecisionTreeClassifier( criterion="entropy")
from sklearn.feature_selection import RFE
rfe = RFE(basetree)

rfe.fit(X,y)

rfe.ranking_

rankdf = pd.DataFrame({"rank" : rfe.ranking_ , "feature":X.columns})

rankdf[rankdf["rank"] == 1]
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , AdaBoostClassifier


X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)

basetree = DecisionTreeClassifier( criterion="gini" , min_samples_split=0.4)
clf = AdaBoostClassifier(n_estimators=50 , learning_rate=0.5)

cross_val_score(clf , X,y,scoring="roc_auc")
