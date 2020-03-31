%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import hdbscan

import time


df = getAllData()
DEPENDENT_VAR = getDependentVariable()
DEPENDENT_VAR in df
"isTraining" in df.columns

bogus_cols =  ["encounter_id" , "patient_id"]
numeric_cols = getNumericColumns(df)

"isTraining" not  in numeric_cols

cat_cols = getCategorialColumns(df)
cat_cols = list(filter(lambda x : x not in bogus_cols + ["hospital_death" , "isTraining"] , cat_cols))
#corr = df.corr()

len(cat_cols + numeric_cols)
df.loc[:,numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
df.loc[:,numeric_cols] = PowerTransformer(method="yeo-johnson").fit_transform(df[numeric_cols])

df = pd.get_dummies(df , columns=cat_cols ,dummy_na=True , drop_first=True)
df =df.drop(bogus_cols , axis=1)

df.shape

#df.describe()

kdf = df.copy()
#kdf = df.loc[:25000 , ]
kdf.loc[:,numeric_cols] = kdf.loc[:,numeric_cols].fillna(0)

distSample = kdf[kdf["isTraining"] == 1].sample(20000)

#kdf = kdf.loc[:10000,]
kdf = kdf.drop("isTraining" , axis=1)
distSample = distSample.drop("isTraining" , axis=1)
#kdf = df.loc[:5000,]
#kdf = df.copy()


knnk = 5

from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores = cpu_count()


from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import wminkowski


def rf(dchunk , start):
    return dchunk.argsort(axis=1)[:,:knnk]


t1 = time.time()
distMat = pairwise_distances_chunked(kdf,  distSample , reduce_func=rf , metric='nan_euclidean',n_jobs=-1, force_all_finite=False)
distMat = list(distMat)[0]
imputeIndices = pd.DataFrame(distMat)
t2 = time.time()

print(t2-t1)

def KNN(kdf):
    #dist = euclidean_distances(kdf , distSample)
    dist = pairwise_distances(kdf, distSample, metric='nan_euclidean', force_all_finite=False)
    indices =  dist.argsort(axis=1)[:,:knnk]
    #kdf = kdf[].fillna()
    return indices


kdf.shape

chunk_size = (int(kdf.shape[0]/30) , kdf.shape[1])


t1 = time.time()
distMat = dd.from_pandas(kdf , chunksize=chunk_size[0]).\
             map_partitions(KNN).compute(scheduler = "processes")

t2 = time.time()
print(t2-t1)

distMat.shape

np.savetxt('test.out', distMat, delimiter=',')
distSample = distSample.fillna(0)
kdf = kdf.fillna(0)
dists = euclidean_distances(kdf , distSample)
dists.shape


imputeIndices = pd.DataFrame(distMat)


sii = imputeIndices.loc[:10000 , ]


df.loc[sii,]
imputeIndices.apply(lambda x: df.loc[x,],axis=1)

def impute(row):
    #i = 0
    #row = sii.loc[i,]
    i = row.name
    rowToImpute = df.loc[i,]
    meanToUse = df.loc[row,].mean(skipna=True)
    nanIndex = rowToImpute.isna() == True
    rowToImpute[nanIndex] = meanToUse[nanIndex]
    return rowToImpute

imputed = imputeIndices.apply(impute,axis=1)

imputed.to_csv("imputed.csv")
