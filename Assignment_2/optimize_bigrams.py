
import matplotlib.pyplot as plt
from ast import Str
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

train = pd.read_csv('Assignment 2/train_bigram.csv')
test = pd.read_csv('Assignment 2/test_bigram.csv')

X_train = train.loc[:, train.columns != 'label']
Y_train = train['label']

X_test = test.loc[:, test.columns != 'label']
Y_test = test['label']

# X_train, X_dev, Y_train, Y_dev = train_test_split(
#     X_train1, Y_train1, test_size=0.1, random_state=42)

# Random Forest
# rf = RandomForestClassifier(random_state=37, max_features='sqrt')
# param_grid_rf = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500], 'max_depth': [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'max_features': ["sqrt", "auto"]}
# clf_rf = GridSearchCV(rf, param_grid=param_grid_rf,
#                       cv=10, verbose=True)
# best_clf_rf = clf_rf.fit(X_train, Y_train)

# print(best_clf_rf.best_estimator_)
# print(best_clf_rf.best_params_)

# # Logstic Regression
# lm = LogisticRegression(penalty='l1', solver='liblinear', random_state=37)
# param_grid = {'C': np.logspace(-4, 4, 20), 'max_iter': [100, 1000, 2500, 5000]}
# clf = GridSearchCV(lm, param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
# best_clf = clf.fit(X_train, Y_train)

# print(best_clf.best_estimator_)
# print(best_clf.best_params_)

dt = DecisionTreeClassifier(random_state=37)
param_grid_dt = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'min_impurity_decrease': [
    0.0, 0.001, 0.005, 0.01, 0.015, 0.02], 'max_features': ['auto', 'sqrt']}
clf_dt = GridSearchCV(dt, param_grid=param_grid_dt,
                      cv=10, verbose=True, n_jobs=-1)
best_clf_dt = clf_dt.fit(X_train, Y_train)

print(best_clf_dt.best_estimator_)
print(best_clf_dt.best_params_)

mb = MultinomialNB()
param_grid_mb = {'alpha': [0.5, 1.0]}
clf_mb = GridSearchCV(mb, param_grid=param_grid_mb,
                      cv=10, verbose=True, n_jobs=-1)
best_clf_mb = clf_mb.fit(X_train, Y_train)

print(best_clf_mb.best_estimator_)
print(best_clf_mb.best_params_)
