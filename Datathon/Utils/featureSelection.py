## Feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier


def RF(X,y,weights={0:1 , 1:9} , n_estimators = 600):
    X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
    rfc = RandomForestClassifier(criterion="entropy" , class_weight=weights, random_state=60 , n_estimators=n_estimators)
    rfc.fit(X_train , y_train)

    model = SelectFromModel(rfc, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)
    print(roc_auc_score(y_test , rfc.predict_proba(X_test)[:,1]))
    return X_new


def adaboost(X,y , n_estimators=100 , learning_rate=1):
    X_train , X_test ,y_train, y_test = train_test_split(X , y , stratify = y)
    clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=0 , learning_rate=learning_rate)
    clf.fit(X, y)
    print(roc_auc_score(y_test , clf.predict_proba(X_test)[:,1]))
    return clf
