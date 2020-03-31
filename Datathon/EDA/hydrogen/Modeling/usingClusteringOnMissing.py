from Datathon.Utils.getData import *

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

df = getTrainingData()

colsWithLargeMissingValues = df.loc[:,df.isna().sum() > df.shape[0] * 0.6]
DEPENDENT_VAR = getDependentVariable()

clusterer = hdbscan.HDBSCAN(min_cluster_size=1000)
#kdf["cluster"]=  clusterer.fit_predict(kdf.iloc[:,:10].isnull())
df["cluster"]=  clusterer.fit_predict(kdf.isnull())
