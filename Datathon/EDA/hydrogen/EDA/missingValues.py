# %%

from Datathon.Utils.getData import DICTDF

from plotnine import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import os

from Datathon.Utils.getData import  *

from scipy.stats import chi2_contingency

from sklearn.cluster import DBSCAN
import hdbscan
# %%


df = getTrainingData()

df.isna().sum()
# %%

# %% markdown
# # Columns Available
# %%
print(list(df.columns))
# %% markdown
# ## Columns that are not a reading
# %%
cols_notareading = [c for c in df.columns if len(re.findall("(min|max|apache)",  c)) == 0]
print(cols_notareading)
# %%
cols_diseases = cols_notareading[cols_notareading.index("aids"):] + ["elective_surgery" ]
print(cols_diseases)
# %% markdown
# # Missing Values
# %% markdown
# These have more than 80 % missing values
# %%

nacounts = df.isna().sum()
cols_extmiss = list(nacounts[nacounts.sort_values() > df.shape[0]*0.8].index)
print(cols_extmiss)

# %% markdown
# # Disease Counts
# %%

tdf = df[cols_diseases + ["encounter_id"]].melt( value_vars =cols_diseases)
tdf["value"].sum()

# %% markdown
# Only 45K rows for these diseases
# Are there any other disease missed
# %%

tdf[tdf["value"] == 0].shape
tdf[tdf["value"] != 0].shape

df[df[cols_extmiss].any(1)].shape


df["icu_stay_type"].unique()

pd.crosstab(df["icu_stay_type"] , df["hospital_death"])


kdf = df
for cold in cols_diseases:
    for colmiss in cols_extmiss:
        ct = pd.crosstab(kdf[cold] , kdf[colmiss].isna())
        c2t = chi2_contingency(ct)
        if c2t[1] < 0.01:
            print(f"{cold} : {colmiss} : {c2t[1]}")

# %%
df[cols_extmiss + ["hospital_death"]]


# %% markdown
# # Which type of columns have most missing Values
# %%
cols_demo = getColumnsByType("demographic")
'icu_admit_type' in df.columns
set(cols_demo).difference(set(df.columns))

cols_demo = list(set(cols_demo).intersection(set(df.columns)))

df[cols_demo].isna().


#### 02/01/20

mdf = df
mdf["id"] = list(df.index)
mdf = pd.melt(mdf , id_vars="id")
mdf["ismissing"] = mdf["value"].isna()
mdf["ismissing"] = mdf["ismissing"].apply(lambda x : 1 if x else 0)
mdf.head()

sns.heatmap(df.isnull())


# %% markdown
# # try clustering rows based on missing Values
# %%

df.isnull().sum()

kdf = df.loc[:,df.isnull().sum() > df.shape[0] * 0.6]
kdf.columns

#clusterer= DBSCAN()
clusterer = hdbscan.HDBSCAN(min_cluster_size=1000)
#kdf["cluster"]=  clusterer.fit_predict(kdf.iloc[:,:10].isnull())
kdf["cluster"]=  clusterer.fit_predict(kdf.isnull())

kdf.shape[0]
clusterMissingSummary = kdf.groupby("cluster").agg(lambda x: x.isna().sum())
clusterMissingSummary = clusterMissingSummary.reset_index()
meltedCluster = pd.melt(clusterMissingSummary , id_vars="cluster")
meltedCluster.head()

(ggplot(meltedCluster , aes(x='factor(cluster)' , y = 'variable' , fill='value'))+ geom_tile())

merged = pd.merge(meltedCluster , DICTDF, left_on="variable" , right_on="Variable Name")
merged.head()

merged.groupby(["cluster" , "Category"]).agg({"value" : "mean"}).reset_index().sort_values("value")

kdf.groupby("cluster").size()

merged.groupby(["cluster"]).apply(lambda x:x[x["value"] == x["value"].max()])


df["missingColumnCluster"] = kdf["cluster"]

perClusterBodyMissingdf = df.groupby(["apache_3j_bodysystem", "missingColumnCluster"]).agg({"encounter_id" : "count"}).reset_index()

(ggplot(perClusterBodyMissingdf)
 + aes(y = 'factor(missingColumnCluster)' ,x ='apache_3j_bodysystem' , fill='encounter_id')
 + geom_tile()
 + theme(axis_text_x = element_text(angle = 65, hjust = 1)))

labCols = getColumnsByType("labs")

hospital_missing_vals = df[["hospital_id" , "encounter_id"] + labCols].groupby("hospital_id").apply(lambda x: x.isna().sum())
hospital_missing_vals["max_missing"] = hospital_missing_vals.mean(axis=1)
hospital_missing_vals["total_encounters"] = df.groupby("hospital_id").agg({"encounter_id" : "count"})
hospital_missing_vals["percent_missing"] = hospital_missing_vals["max_missing"]/hospital_missing_vals["total_encounters"]
hospital_missing_vals[["percent_missing"]].sort_values("percent_missing" , ascending=False)
