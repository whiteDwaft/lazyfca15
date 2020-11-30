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


class MyClassifier():
    def __init__(self, min_supp=0.9, min_similarity=0.7):
        self.min_supp = min_supp
        self.min_similarity = min_similarity

    def __predict_one(self, x):
        pos_intersection = (self.pos * x)
        pos_conf = (pos_intersection == x).sum(axis=1) / x.shape[0]
        neg_intersection = (self.neg * x)
        neg_conf = (neg_intersection == x).sum(axis=1) / x.shape[0]

        pos_intersection = pos_intersection[pos_conf >= min(self.min_similarity, pos_conf.max())]
        neg_intersection = neg_intersection[neg_conf >= min(self.min_similarity, neg_conf.max())]

        pos_dash = (pos_intersection.dot(pos_intersection.T) ==
                    pos_intersection.sum(axis=1).reshape(pos_intersection.shape[0], 1))
        pos_dash = pos_dash.sum(axis=1) / self.pos.shape[0]
        pos_dash = pos_dash[pos_dash >= min(self.min_supp, pos_dash.max())]

        neg_dash = (neg_intersection.dot(neg_intersection.T) ==
                    neg_intersection.sum(axis=1).reshape(neg_intersection.shape[0], 1))
        neg_dash = neg_dash.sum(axis=1) / self.neg.shape[0]
        neg_dash = neg_dash[neg_dash >= min(self.min_supp, neg_dash.max())]

        pos_coeff = pos_dash.mean()
        neg_coeff = neg_dash.mean()

        c = 1 / (pos_coeff + neg_coeff)
        return [c * neg_coeff, c * pos_coeff]

    def fit(self, X, y):
        self.pos, self.neg = X[y == 1], X[y == 0]
        return self

    def predict(self, X):
        return [np.argmax(l) for l in self.predict_proba(X)]

    def predict_proba(self, X):
        return np.array([self.__predict_one(x) for x in X])

    def get_params(self, deep=True):
        return {'min_supp': self.min_supp, 'min_similarity': self.min_similarity}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


skf = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True)
clf = MyClassifier()
gs = sklearn.model_selection.RandomizedSearchCV(clf, {'min_supp': np.linspace(0, 1, 20),
                                                  'min_similarity': np.linspace(0.7, 1, 20)},
                                            scoring='accuracy', n_jobs=-1, n_iter=100, cv=skf, error_score=0)
res = gs.fit(X_train, y_train)
print(res.best_params_, res.best_score_)
