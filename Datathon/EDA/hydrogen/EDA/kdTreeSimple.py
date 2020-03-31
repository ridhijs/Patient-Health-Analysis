%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *
from sklearn.neighbors import KDTree , BallTree
from sklearn.preprocessing import StandardScaler

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from random import sample

df = getAllData()
numeric_cols = getNumericColumns(df)
diag_cols = getDiagnosisColumns()

kdf = df[numeric_cols + diag_cols]

kdf.loc[:,numeric_cols] = StandardScaler().fit_transform(kdf[numeric_cols])
#kdf.loc[:,numeric_cols]  = kdf.loc[:,numeric_cols].round(2)
#kdf.loc[:,numeric_cols]  = kdf.loc[:,numeric_cols] * 100


#sample(list(numeric_cols) , 10) + sample(list(diag_cols) , 10)
kdf = pd.get_dummies(kdf , columns=diag_cols, dummy_na=True)
kdf.shape

#tree = KDTree(kdf, leaf_size=15)

#tree = cKDTree(kdf.values , leafsize = 400 )
tree = BallTree(kdf, leaf_size = 400 )

%timeit  tree.query(kdf.loc[: 100, ].fillna(0) , 1)

dist , i = tree.query(kdf.loc[: , ].fillna(0) , 5)

imputeDf = kdf.copy()


#imputeDf = imputeDf.loc[50001:100000,]
imputeDf = imputeDf.loc[:100,]
npartitions = 10

import numba

@numba.jit(parallel = True)
def imputeRow(row):
    #row = kdf.loc[1:1,]
    i = row.name
    dist , neighbors = tree.query([row.fillna(0)], 5)
    meanVals = kdf.loc[neighbors[0] , ].astype("float").mean(skipna= True)
    naCols = row.index[row.isna()]
    meanVals = meanVals[naCols]
    #naCols = list(naCols)[0]
    #print(naCols)
    #row.update( pd.Series(meanVals[naCols] , index=naCols) )
    imputeDf.loc[i,naCols] = meanVals[naCols]
    return imputeDf.loc[i,:]


numbaFillDf = imputeDf.apply(imputeRow , axis=1)

%timeit imputeDf.apply(imputeRow , axis=1)
#def imputeRowNumba(rowArr , )
numbaFillDf = imputeDf.apply(imputeRow , axis=1)


import dask.dataframe as dd
from dask.multiprocessing import get


fidf = dd.from_pandas(imputeDf , npartitions=npartitions).\
         map_partitions(lambda df : df.apply(imputeRow , axis=1)).\
         compute(scheduler="processes")

fidf.to_csv("imputed_50001.csv")
