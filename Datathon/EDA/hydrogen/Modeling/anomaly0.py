%load_ext autoreload
%autoreload 2
%matplotlib inline

from Datathon.Utils.apache2Calc import *
from Datathon.Utils.getData import *
from Datathon.Utils.pipeFunctions import *
from sklearn.preprocessing import StandardScaler
from plotnine import *
#df = getTrainingData()

adf = getTrainingData()

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from pyod.models.sos import SOS
from pyod.models.knn import KNN
from pyod.models.so_gaal import SO_GAAL
from pyod.models.lscp import LSCP
from pyod.models.cblof import CBLOF
from pyod.utils.utility import get_label_n
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

import logging

logging.basicConfig(level=logging.INFO)
# class TestT:
#     def __init__(self):
#         self.dd = None
#
#
# self = TestT
# isTraining = True
#
# def _toXY(df):
#     y = df[self.DEPENDENT_VARIABLE]
#     X = df.drop([self.DEPENDENT_VARIABLE] , axis=1)
#     return X,y
#
# self._toXY = _toXY

class Pipeline:

    def _toXY(self,df):
        y = df[self.DEPENDENT_VARIABLE]
        X = df.drop([self.DEPENDENT_VARIABLE] , axis=1)
        return X,y

    def initialTransform(self,isTraining=True):
        ohe = self.ohe
        func = 'fit_transform' if isTraining else 'transform'
        self.df.loc[:, self.numeric_cols] = getattr( self.num_mean, func)( self.df.loc[:,self.numeric_cols])
        self.df.loc[:, self.cat_cols] = getattr( self.cat_freq, func)( self.df.loc[:,self.cat_cols])
        self.df.loc[:, self.numeric_cols] = getattr( self.rs, func)( self.df.loc[:,self.numeric_cols])
        self.df.loc[:, self.numeric_cols] = getattr( self.pt, func)( self.df.loc[:,self.numeric_cols])
        ohedf =  getattr(self.ohe, func)(self.df[self.cols_to_dummy])
        self.df = pd.DataFrame(ohedf).join(self.df)

        self.df = self.df.drop(self.cols_to_dummy, axis=1)
        self.X ,self.y = self._toXY(self.df)


    def outlierReimputation(self, isTraining=True):
        func = 'fit_transform' if isTraining else 'transform'
        if isTraining:
            self.outlierKNN.fit(self.X)
        preds = self.outlierKNN.predict(self.X)

        if isTraining:
            alive_outlier = np.logical_and(preds== 1  , self.y == 0)
            alive_normal = np.logical_and(preds== 0  , self.y == 0)
            dead_normal = np.logical_and(preds== 0  , self.y == 1)
            dead_outlier = np.logical_and(preds== 1  , self.y == 1)
        else:
            alive_outlier = []
            dead_outlier = []
            dead_normal = preds == 0
            dead_outlier = preds == 1


        ids = [alive_normal , alive_outlier , dead_normal , dead_outlier]

        impute_again_df = self.adf.copy()
        #i = 0
        #ids = alive_normal
        for i,ids in enumerate([alive_normal , alive_outlier , dead_normal , dead_outlier]):
            #print(np.where(ids == True)[0].shape)
            num_mean = self.num_means[i]
            cat_freq = self.cat_freqs[i]
            indices = np.where(ids == True)[0]
            if len(indices) > 0:
                impute_again_df.loc[indices,self.numeric_cols] = getattr(num_mean, func)(impute_again_df.loc[indices,self.numeric_cols])
                impute_again_df.loc[indices,self.cat_cols] = getattr(cat_freq, func)(impute_again_df.loc[indices,self.cat_cols])

        self.impute_again_df = impute_again_df

    def laterEncodingAndTransformations(self , isTraining=True):
        func = 'fit_transform' if isTraining else 'transform'
        impute_again_df = self.impute_again_df.drop(self.cat_cols_useless , axis=1)
        impute_again_df.loc[:,self.numeric_cols] = getattr(self.later_num_transform , func)(impute_again_df.loc[:,self.numeric_cols])
        impute_again_df.loc[:,self.cat_cols_minus_useless] = impute_again_df.loc[:,self.cat_cols_minus_useless].apply(lambda x: getattr(self.label_encoders[x.name] , func)(x))
        self.impute_again_df = impute_again_df
        self.X , self.y = self._toXY(self.impute_again_df)
        return self.X , self.y

    def GetTransformedData(self , isTraining):
        logging.info("Starting- Initial Transform")
        self.initialTransform(isTraining)
        logging.info("Starting- Outlier Re imputation")
        self.outlierReimputation(isTraining)
        logging.info("Starting- Label Encoding")
        return self.laterEncodingAndTransformations(isTraining)


    def fit(self,df):
        logging.info("Initializaing Pipeline")
        isTraining = True
        self.adf = df
        self.df = df.copy()

        self.numeric_cols = getNumericColumns(df)
        self.cat_cols = getCategorialColumns(df)
        self.DEPENDENT_VARIABLE = getDependentVariable()

        self.cat_cols_useless =  [ "encounter_id" , "hospital_id" , "patient_id" , "icu_id"]
        self.cat_cols_minus = [c for c in self.cat_cols if c not in ["clusterId","hospital_death", "encounter_id" , "hospital_id" , "patient_id"]]
        self.cat_cols_minus_useless = [c for c in self.cat_cols if c not in ["clusterId", "encounter_id" , "hospital_id" , "patient_id" , "icu_id" ]]
        self.cols_to_dummy = [c for c in self.cat_cols_minus_useless if c != "hospital_death"]

        self.num_mean = SimpleImputer(strategy="median")
        self.cat_freq = SimpleImputer(strategy="most_frequent")
        self.rs = RobustScaler()
        self.pt = PowerTransformer()
        self.ohe = OneHotEncoder(handle_unknown='ignore' , sparse=False)
        self.outlierKNN = KNN()

        self.num_means = [SimpleImputer(strategy="median") for i in range(4)]
        self.cat_freqs = [SimpleImputer(strategy="most_frequent") for i in range(4)]
        self.label_encoders = defaultdict(LabelEncoder)
        self.later_num_transform = PowerTransformer()
        return self.GetTransformedData(isTraining)

    def transform(self, df):
        isTraining = False
        self.adf = df
        self.df = df.copy()
        return self.GetTransformedData(isTraining)


pipe = Pipeline()
X,y = pipe.fit(adf)


#X,y = pipe.X , pipe.y
# def godMode(df):
#     idf = df.copy()
#     #print(set(df.columns).difference(set(adf.columns)))
#
#     ndf = initialTransform(idf ,num_mean,cat_freq,rs,pt,ohe)
#
#     y = ndf[DEPENDENT_VARIABLE]
#     X = ndf.drop([DEPENDENT_VARIABLE],axis=1)
#
#     getattr(outlierKNN , 'fit' if isTraining else 'predict')(X)
#
#     impute_again_df = outlierReimputation(outlierKNN , adf, X,y)
#
#     ready_df = laterEncodingAndTransformations(impute_again_df)
#
#     y = ready_df[DEPENDENT_VARIABLE]
#     X = ready_df.drop([DEPENDENT_VARIABLE] , axis=1)
#     return X,y
#
#
# X,y = godMode(df)
#### Modelling
import lightgbm as lgb

params = {
  'max_depth': 10,
  'n_estimators ': 1000,
  'objective': 'binary',
  'colsample_bytree': 0.2,
  "class_weight":{0:1 , 1:200},
  "base_score":0.18,
  "n_jobs":-1,
  "metric":"auc",
  "reg_alpha":250,
  "reg_lambda":500,
}

lgclf = lgb.LGBMClassifier(**params)


from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow import keras


def create_keras_model(dim):

    def cl():
        model = keras.Sequential()
        model.add(keras.layers.Dense(64,activation='relu', input_dim=dim))
        model.add(keras.layers.Dense(10,activation='sigmoid'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        model.compile(loss=keras.losses.BinaryCrossentropy(),optimizer=keras.optimizers.Adam(lr=1e-3), metrics=[ keras.metrics.AUC(name='auc')])
        return model

    return cl


build_fn = create_keras_model(X.shape[1])
kerasclf = KerasClassifier(build_fn,verbose=0)

#model = build_fn()

from sklearn.model_selection import learning_curve


train_sizes, train_scores, valid_scores = learning_curve(kerasclf, X, y,scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 10), cv=2)

lcurveplotdf = pd.DataFrame({"train_size":train_sizes , "train_score" : train_scores[:,1] , "valid_score":valid_scores[:,1]})

ggplot(lcurveplotdf ) + \
    geom_line(aes(x="train_size" , y="train_score") , color="red") + \
    geom_line(aes(x="train_size" , y="valid_score") , color="green")




tdf = getUnlabledData()

seri = tdf["encounter_id"]

def replaceUnseenCategories(seri):
    unq = seri.unique()
    seen =  adf[seri.name].unique()
    most_freq = adf[seri.name].value_counts().index[0]
    for v in unq:
        #print(v)
        if v not in seen:
            seri = seri.replace(v ,most_freq )
    return seri

tdf.loc[:,pipe.cat_cols] = tdf[pipe.cat_cols].apply(replaceUnseenCategories )

# len(pipe.numeric_cols)
# set(getNumericColumns(adf)).difference(set(pipe.numeric_cols))

Xt,yt = pipe.transform(tdf)
#Xt,yt = pipe.X,pipe.y
kerasclf.fit(X,y)





preds = kerasclf.predict_proba(Xt)

saveKeraspreds  =preds[:,1]
results = pd.DataFrame({"encounter_id" : seri , "hospital_death" :preds[:,1] })
results.to_csv("./submission.csv" , index=False)
