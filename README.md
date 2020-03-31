### WIDS Hackathon 2020

* **Last Sumission Model Summary**
  - /Datathon/EDA/hydrogen/EDA/zoomFeatureImportance.py
  - **ROC_AUC - .89768**
  ### Pipeline >
  - **Missing Values**
    - num_mean = SimpleImputer(strategy="mean")
    - cat_freq = SimpleImputer(strategy="most_frequent")
  - **Add Apache Score column** - getAPACHEScore
  - **Feature Transformations**
   - rs = RobustScaler()
   - pt = PowerTransformer()
  - **Feature Reduction**
    - fa = FeatureAgglomeration(n_clusters=None,affinity="precomputed",compute_full_tree=True, linkage="average" ,distance_threshold=1/corrcoefmin)
  - **Categorical Columns**
    - ohe = OneHotEncoder(sparse=False , handle_unknown='ignore')
  - **Modelling**
    - import lightgbm as lgb
    - 
    ```
    params = {
      'max_depth': 10,
      'n_estimators ': 10,
      'objective': 'binary',
      'colsample_bytree': 0.8,
      "class_weight":{0:1 , 1:20},
      "base_score":0.2,
      "n_jobs":-1,
      "metric":"auc",
      "reg_alpha":0.4,
      "reg_lambda":0.18,
    }
    ```

* Directory Structure
  * **/Datathon/**
    - **/Data/**
      - Houses the data
      - This data can be pulled following the /Documentations/Setup.md
        - Or By manually downloading all files from [Kaggle](https://www.kaggle.com/c/widsdatathon2020/data) and placing them in the folder
    - **/Utils**
      - getData.py
        - It has util functions for loading the data dictionary , reading training data into pandas data frame , loading up the unlabeled/test data etc
      - apache2Calc.py
        - This allows to create a new column for apache 2 score based on other columns provided

    - **/EDA/hydrogen/EDA**
      - afterImpute.py (1)
        - This file relies on having imputed the data
        - it then
          1. loads the data
          2. **DropCorrelated** - drops highly correlated columns
          3. **ReplaceColumnsWithIsMissing** - adds ismissing column for features which still have missing values
          4. **PowerTransformer** - runs power transformer to normalise the numerical columns
          5. **get_dummies** - Dummy Codes the categorical Columns
          6.  Then runs various **different models**
            1. adaboost
            2. DecisionTreeClassifier
            3. GradientBoostingClassifier
            4. RandomizedSearchCV
            5. StackingClassifier with
              1. RandomForestClassifier
              2. AdaBoostClassifier
      - clusteringImputation.py (2)
        - This is one of the methods tried to **impute** the missing values
        - It clusters the rows using euclidean_distances and then imputes using the mean of the clusters -```def imputeRow(row):```
      - customKNN.py (3)
        - This file tries a custom implementation of KNN for imputation
        - This was done because the knn imputer implementation of sklearn holds everything in memory which needs around **13.7 gb** to hold the distance calculations
      - dataDistributions.py (4)
        - this was an eda file that looks at the different columns and their density distributions
      - diagnosis_eda.py (5)
        - This file tries to explore the diagnosis and apache columns in the data to see if there are any discernible patterns
      - feature_agglomeration.py (6)
        - this was an attempt **feature reduction by clustering** along features and then use only one representative from the cluster
        - distance metric used was **correlation**
      - featureSelection.py (7)
        - This file tries to use **LogisticRegressionCV** and LinearSVC along with **SelectFromModel** to try and reduce features
      - kdTree.py (8)
        - This was attempted along with (3) to reduce memory requirements
        - Uses **Numba , Dask**

      - missingValues.py (9)
        - This was an attempt in conjunction with (2), to explore the missingness in different clusters and columns
      - NNEt.py (10)
        - This was an attempt to apply **Neural Networks** to the imputed Data
      - pca.py (11)
        - This was an attempt at feature reduction using **PCA**
      - xgboost.py (12)
        - This was an attempt to aaply **Xgboost** and search for the optimal hyper parameters
