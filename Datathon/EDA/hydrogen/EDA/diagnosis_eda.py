from Datathon.Utils.getData import getTrainingData,getColumnsByType

from sklearn.linear_model import  LogisticRegression

import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns

df = getTrainingData()

df.head()

df["hospital_death"].value_counts()

df["apache_3j_diagnosis"].value_counts()

cst = pd.crosstab(df["apache_3j_diagnosis"] ,df["hospital_death"]).sort_values(1 , ascending=False)

cst


df.loc[df["apache_3j_diagnosis"].isin([102.01 , 501.05]),  "apache_2_diagnosis"].value_counts()

df["apache_2_diagnosis"].value_counts().reset_index()

matplotlib.rcParams['figure.figsize'] = [12, 10]

sns.barplot(orient="h",data = df["apache_2_diagnosis"].value_counts().reset_index() , y= "index" , x="apache_2_diagnosis")


pd.crosstab(df["apache_2_diagnosis"] ,df["hospital_death"]).sort_values(1 , ascending=False)

#chi2_contingency(cst)
labcols = getColumnsByType("labs")

apacheCols = getColumnsByType("APACHE covariate")

df[apacheCols].describe()
tdf = df.dropna()
for c in labcols:
    model = LogisticRegression()
    model.fit(tdf[apacheCols] , tdf[c].isna())
    print(model)
    # cst = pd.crosstab(df[c].isna() , df["apache_2_diagnosis"])
    # _,pval , _ , _ = chi2_contingency(cst)
    # if pval < 0.05:
    #     print(f"{c} - {pval}")


df["apache_4a_hospital_death_prob"]

sns.boxplot(data=df , y="apache_4a_hospital_death_prob" , x="hospital_death")


sns.boxplot(data=df , y="apache_4a_icu_death_prob" , x="hospital_death")
