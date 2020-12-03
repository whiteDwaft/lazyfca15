import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection

columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square',
           'middle-middle-square', 'middle-right-square', 'bottom-left-square',
           'bottom-middle-square', 'bottom-right-square', 'Class']
data = pd.read_csv("tic-tac-toe.data", header=None)
data.columns = columns
data.Class = data.Class.map(lambda x: 1 if x == 'positive' else 0)
data.to_csv('tic-tac-toe.csv', index=False)

data = pd.get_dummies(data)
X = data.iloc[:, 1:].values
print(type(X))
y = data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)




def TP(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[0,0]

def TN(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[1,1]

def FP(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[1,0]

def FN(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[0,1]

def TPR(y_true, y_pred):
    return TP(y_true, y_pred)/(TP(y_true, y_pred) + FN(y_true, y_pred))

def TNR(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FP(y_true, y_pred))

def NPV(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FN(y_true, y_pred))

def FPR(y_true, y_pred):
    return FP(y_true, y_pred)/(FP(y_true, y_pred) + TN(y_true, y_pred))

def FDR(y_true, y_pred):
    return FP(y_true, y_pred)/(FP(y_true, y_pred) + TP(y_true, y_pred))

def TNR(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FP(y_true, y_pred))

metrics = [TP, TN, FP, FN, TPR, TNR, NPV, FPR, FDR, TNR,
           sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
           sklearn.metrics.recall_score, sklearn.metrics.roc_auc_score, sklearn.metrics.f1_score]
metrics_names = [func.__name__ for func in metrics]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

for metric_name, metric in zip(metrics_names, metrics):
    score = metric(y_test, y_pred)
    print(metric_name, '=', score)

# clf = gs.best_estimator_
# # clf = MyClassifier(min_similarity=0.9, min_supp=0.4)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# for metric_name, metric in zip(metrics_names, metrics):
#     score = metric(y_test, y_pred)
#     print(metric_name, '=', score)
#
# skf = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True)
# scores = sklearn.model_selection.cross_val_score(clf, X, y, scoring='accuracy', n_jobs=-1, cv=skf)
# print('Accuracy', scores.mean())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

for metric_name, metric in zip(metrics_names, metrics):
    score = metric(y_test, y_pred)
    print(metric_name, '=', score)