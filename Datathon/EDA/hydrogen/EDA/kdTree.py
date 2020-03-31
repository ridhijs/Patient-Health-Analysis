%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *
from Datathon.Utils.getPreparedData import *

#df.describe()

df = getPreparedAllData()


#wonadf = df.dropna()
# wonadf.shape

numeric_cols = getNumericColumns(df)
cat_cols = getCategorialColumns(df)
kdf = df.copy()
#kdf = df.loc[:25000 , ]
kdf.loc[:,numeric_cols] = kdf.loc[:,numeric_cols].fillna(0)

distSample = kdf[kdf["isTraining"] == 1].sample(20000)
kdf = kdf.drop("isTraining" , axis=1)
distSample = distSample.drop("isTraining" , axis=1)

import scipy.spatial.distance
from sklearn.neighbors import KDTree
#kdf = kdf.loc[:10000,]

from scipy.spatial.distance import cdist


from numba import jit

@jit(nopython=True)
def pairwise_python(X,Y):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - Y[j, k]
                d += tmp * tmp
            D[i, j] = d
    return D

X = kdf.loc[:500,]
X = X.to_numpy(np.float64)

X.shape
%timeit pairwise_python( X, X)


tree = KDTree(kdf, leaf_size=15)

row = kdf.loc[20:21,]
%timeit tree.query(row , 5)
