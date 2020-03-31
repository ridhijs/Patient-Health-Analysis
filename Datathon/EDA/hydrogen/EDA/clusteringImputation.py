%load_ext autoreload
%autoreload 2

from Datathon.Utils.getData import *
from sklearn.neighbors import KDTree , BallTree
from sklearn.preprocessing import StandardScaler

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from random import sample

from sklearn.cluster import KMeans

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

ndf = kdf.copy()
ndf = ndf.fillna(-4)

clusters = KMeans(n_clusters = 50).fit(ndf)

ndf["clusterId"] = clusters.predict(ndf)
ndf["clusterId"].value_counts()

kdf["clusterId"] = ndf["clusterId"]
clusters.score()

meansDf = kdf.groupby("clusterId").mean()

imputeDf = kdf.copy()
imputeDf = imputeDf.loc[:10000,]

imputeDf.shape
import numba

@numba.jit
def imputeRow(row):
    #row = kdf.loc[1:1,]
    i = row.name
    naCols = row.index[row.isna()]
    meanVals = meansDf.loc[row["clusterId"] ,naCols ].astype("float")
    row.update(meanVals[naCols])
    return row


def imputeRowWONumba(row):
    i = row.name
    naCols = row.index[row.isna()]
    meanVals = meansDf.loc[row["clusterId"] ,naCols ].astype("float")
    row.update(meanVals[naCols])
    return row



#allImputed = imputeDf.apply(imputeRow , axis=1)
%timeit imputeDf.apply(imputeRow , axis=1)
%timeit imputeDf.apply(imputeRowWONumba , axis=1)


imputedFDf = imputeDf.apply(imputeRowWONumba , axis=1)
imputedFDf.shape
imputedFDf.to_csv("impute_complete.csv" ,index=False)

imputedFDf.columns
