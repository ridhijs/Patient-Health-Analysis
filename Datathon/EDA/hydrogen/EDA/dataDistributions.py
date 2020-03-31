from Datathon.Utils.getData import getTrainingData,getColumnsByType , getDependentVariable

from Datathon.Utils.getData import *

from plotnine import *

import pandas as pd
import pandas.api.types as ptypes
import numpy as np

from sklearn.preprocessing import PowerTransformer

df = getTrainingData()
df


# Transformations needed
# Age needs log(-x)
# bmi log(x)
# height ? different modals. ethnicities ? gender ? can it be normal ? galton's quincunx says normal independent components overall normalize
# drop readmission_status ? only 0s
# weight log(x)
#

depVar = getDependentVariable()
for col in df.columns :
    if ptypes.is_numeric_dtype(df[col]) and col.find("_id") < 0 and col != depVar:
        kdf = df[[col , depVar]]
        kdf = kdf.dropna()
        p = (ggplot(kdf) + aes(x= col , fill=f"factor({depVar})") + geom_histogram())
        print(p)



(ggplot(df) + aes(x= 'np.log(bmi)') + geom_histogram())
