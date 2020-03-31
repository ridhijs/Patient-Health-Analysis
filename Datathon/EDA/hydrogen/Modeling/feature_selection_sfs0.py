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


from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2 , mutual_info_classif
#
# cat_cols_minus = [c for c in cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
# cat_cols_minus_useless = [c for c in cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
# cols_to_dummy = [c for c in cat_cols_minus_useless if c != "hospital_death"]
#
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(handle_unknown='ignore' , sparse=False)
# ohedf = ohe.fit_transform(df[cols_to_dummy])
#
# ndf = pd.DataFrame(ohedf).join(df)

numeric_cols

DEPENDENT_VARIABLE = getDependentVariable()
y = df[DEPENDENT_VARIABLE]
#X = ndf.drop([DEPENDENT_VARIABLE] + cat_cols_minus_useless,axis=1)
X = df[numeric_cols]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import logging

logging.basicConfig(level=logging.INFO)

class CustomSequentialFeatureSelection:

    def __init__(self , estimator,leave_out , max_iter):
        self.estimator = estimator
        self.leave_out  =leave_out
        self.max_iter = max_iter

    def _fit(self,X,y,leave_out=1):
        self.estimator.fit(X,y)
        selected = np.argsort(self.estimator.feature_importances_)[leave_out:]
        #print(self.estimator.feature_importances_)
        return X.iloc[:,selected].columns

    def fit(self, X,y):
        X_train , X_test , y_train , y_test = train_test_split(X,y , stratify=y)
        results = pd.DataFrame({"train_auc" : [] , "test_auc" : [] ,"num_features" : []})
        selected_features = X_train.columns
        fitnessScore = pd.DataFrame({"count" :np.zeros(len(selected_features))} , index=selected_features)
        i = 0
        while True:
            if self.max_iter != -1 and i >self.max_iter:
                break

            i += 1

            _x_train = X_train[selected_features]
            _x_test = X_test[selected_features]
            logging.info(f"Iter : {i} Started")
            selected_features = self._fit(_x_train,y_train , self.leave_out)
            fitnessScore.loc[selected_features,:] += 1

            if len(selected_features) < 1:
                break

            #selected_features = list(_x_train.iloc[:selected_features].columns)
            auc_train = roc_auc_score(y_train , self.estimator.predict_proba(_x_train)[:,1])
            auc_score = roc_auc_score(y_test , self.estimator.predict_proba(_x_test)[:,1])
            results = pd.concat([results , pd.DataFrame({"train_auc" : [auc_train] , "test_auc" : [auc_score] ,"num_features" : [len(selected_features)]})])
            logging.info(f"\nIter : {i} : AUC : {auc_train} - {auc_score} Selected Features : ({len(selected_features)})")

        self.selected_features = selected_features
        return results,fitnessScore

dtc = RandomForestClassifier(max_depth=3,min_samples_split=0.1)
csfs = CustomSequentialFeatureSelection(dtc , 4, -1)
results,fitnessScores = csfs.fit(X_train,y_train)

fitnessScores.sort_values("count" , ascending=False).head(50)
'apache_4a_hospital_death_prob' in csfs.selected_features

from plotnine import *

ggplot(results ) + \
 geom_line(aes(x = "num_features" , y="test_auc") ,color="green") + \
 geom_line(aes(x = "num_features" , y="train_auc") ,color="red") + \
 geom_hline(aes(yintercept = 1) ,color="blue") + \
 geom_hline(aes(yintercept = 0.5) ,color="blue")
