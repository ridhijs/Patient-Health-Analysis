%load_ext autoreload
%autoreload 2
%matplotlib inline

from Datathon.Utils.apache2Calc import *
from Datathon.Utils.getData import *
from Datathon.Utils.pipeFunctions import *

from sklearn.pipeline import Pipeline


from category_encoders.woe import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pyod.models.knn import KNN

from sklearn.preprocessing import FunctionTransformer

class RemoveOutliers():

    def __init__(self):
        self.estimator = KNN()

    def _remove(self,X):
        preds = self.estimator.predict(X)
        return X[preds , :]

    def fit(self,X,y=None):
        self.estimator.fit(X)
        return self

    def transform(self,X,y=None):
        return self._remove(X)

adf = getTrainingData()
#adf = adf.dropna(thresh=adf.shape[0]*0.7 , axis=1)

cols_useless =[ "encounter_id", "hospital_id","patient_id" , "icu_id" , "gender" , "ethnicity"]
#adf = adf.drop(cols_useless,  axis=1)

#cat_cols = [c for c in getCategorialColumns(adf) if c not in cols_useless + ["hospital_death"]]
def tooManyMissing(adf):
    tdf = adf.dropna(thresh=adf.shape[0]*0.75 , axis=1)
    return list(set(adf.columns).difference(set(tdf.columns)))

tooManyMissing_cols = tooManyMissing(adf)

cat_cols = [c for c in getCategorialColumns(adf) if c not in ["hospital_death"] + tooManyMissing_cols]
num_cols = [c for c in getNumericColumns(adf) if c not in tooManyMissing_cols]
DEPENDENT_VARIABLE = getDependentVariable()

numeric_cols_pipe = Pipeline(steps=[
    ('mean impute' ,SimpleImputer(strategy="mean") )
    ,('Standard Scale' ,StandardScaler() )
])


cat_cols_pipe = Pipeline(steps=[
    ('Drop Useless' ,
        ColumnTransformer([
            ('Drop Useless Categorical Features' ,'drop' ,cols_useless )
        ],remainder='passthrough')
    )
    ,('Most Frequent impute' ,SimpleImputer(strategy="most_frequent"))
    ,('Encoding' , WOEEncoder())
])

transform_pipe = Pipeline(steps=[
     ('Column Transform' ,
        ColumnTransformer([
            ('Drop Too Many Missing' , 'drop' , tooManyMissing_cols)
            ,('Cat Cols Pipe' ,cat_cols_pipe ,cat_cols )
            ,('Numeric Cols Pipe' ,numeric_cols_pipe ,num_cols )
        ])
    )
    #,('Drop Outlier Rows' ,RemoveOutliers())
    ,('Scale Everything' , StandardScaler())

])

X = adf.drop([DEPENDENT_VARIABLE] , axis=1)
y = adf[DEPENDENT_VARIABLE]

trX = transform_pipe.fit_transform(X,y)

#trX = RemoveOutliers().fit(trX).transform(trX)
trX.shape


import lightgbm as lgb

params ={'n_estimators':50000,
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    #'is_unbalance' : True ,
                    'scale_pos_weight' : 3
                    ,"n_jobs":2
                    }

lgclf = lgb.LGBMClassifier(**params)


from plotnine import *
from sklearn.model_selection import learning_curve


train_sizes, train_scores, valid_scores = learning_curve(lgclf, trX, y,scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

lcurveplotdf = pd.DataFrame({"train_size":train_sizes , "train_score" : train_scores[:,1] , "valid_score":valid_scores[:,1]})

ggplot(lcurveplotdf ) + \
    geom_line(aes(x="train_size" , y="train_score") , color="red") + \
    geom_line(aes(x="train_size" , y="valid_score") , color="green")


lgclf.fit(trX,y)
testdf = getUnlabledData()

Xtb = testdf.drop([DEPENDENT_VARIABLE] , axis=1)
ytb = testdf[DEPENDENT_VARIABLE]

testX = transform_pipe.transform(Xtb)

testX.shape
preds = lgclf.predict_proba(testX)

saveKeraspreds  =preds[:,1]
results = pd.DataFrame({"encounter_id" : testdf['encounter_id'] , "hospital_death" :preds[:,1] })
results.to_csv("./submission.csv" , index=False)
