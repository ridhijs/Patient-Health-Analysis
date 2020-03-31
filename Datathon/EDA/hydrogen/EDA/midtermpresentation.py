from Datathon.Utils.getData import DICTDF
from Datathon.Utils.getData import  *
from plotnine import *


df = getTrainingData()

df.shape


DICTDF.groupby("Category").count()

bgColorHex = "#25292f"
mt = theme(panel_grid_major = element_blank()
                 , panel_grid_minor = element_blank()
                 ,panel_background = element_rect(fill = bgColorHex)
                 ,plot_background  = element_rect(fill =bgColorHex, colour = 'white')
                 ,axis_text = element_text( face = "bold", colour = "white",size = 8)
                 ,axis_title = element_text( face = "bold", colour = "white",size = 10)
                 ,legend_background = element_rect(fill =bgColorHex, colour = 'white'))



df["hospital_death"].value_counts()

ggplot(df , aes(x="hospital_death")) + geom_bar() + mt


missingCount = (df.isna().sum().sort_values(ascending=False).head(100) / df.shape[0])*100
missingCountDf = pd.DataFrame({"counts" : missingCount.values , "col" : missingCount.index})

ggplot(missingCountDf ,aes(y="counts" , x="col")) + geom_col() + coord_flip() + labs(y="% Missing Values")+ mt





for col in getNumericColumns(df)[:10]:
    p = ggplot(df , aes(x=col)) + geom_density(color="white") + mt
    print(p)


# Baseline
DEPENDENT_VAR = getDependentVariable()
catcols = getCategorialColumns(df)
catcolsWoBogus = [c for c in catcols if c not in ["hospital_id" , "encounter_id" , "icu_id" , "patient_id"]]
catcolsWoBogusWoTarget = [c for c in catcolsWoBogus if c != DEPENDENT_VAR]
from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy="most_frequent")
woTarget = df.drop([DEPENDENT_VAR] , axis=1)
df.loc[:,woTarget.columns] = si.fit_transform(woTarget)
ndf = pd.get_dummies(df , columns=catcolsWoBogusWoTarget , drop_first=True)
ndf.shape
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.model_selection import cross_val_score
cross_val_score(rfc , ndf.drop(DEPENDENT_VAR,axis=1) , df[DEPENDENT_VAR] , scoring="roc_auc")
