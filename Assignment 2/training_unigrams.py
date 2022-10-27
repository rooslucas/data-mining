
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

train = pd.read_csv('Assignment 2/train_unigram.csv')
test = pd.read_csv('Assignment 2/test_unigram.csv')

X_train = train.loc[:, train.columns != 'label']
Y_train = train['label']

X_test = test.loc[:, test.columns != 'label']
Y_test = test['label']

print("Unigrams:")

rf = RandomForestClassifier(
    n_estimators=100, max_depth=7, random_state=37, max_features='sqrt')
rf.fit(X_train, Y_train)
print(f"RF train score: {rf.score(X_train, Y_train)}")
print(f"RF test score: {rf.score(X_test, Y_test)}")

lr = LogisticRegression(C=0.615848211066026, penalty='l1',
                        solver='liblinear', random_state=37, max_iter=100)
lr.fit(X_train, Y_train)
print(f"LR train score: {lr.score(X_train, Y_train)}")
print(f"LR test score: {lr.score(X_test, Y_test)}")


dt = DecisionTreeClassifier(
    max_depth=15, max_features='auto', min_impurity_decrease=0.001, random_state=37)
dt.fit(X_train, Y_train)
print(f"DT train score: {dt.score(X_train, Y_train)}")
print(f"DT test score: {dt.score(X_test, Y_test)}")

clf = MultinomialNB(alpha=0.5)
clf.fit(X_train, Y_train)
print(f"MB train score: {clf.score(X_train, Y_train)}")
print(f"MB test score: {clf.score(X_test, Y_test)}")

print(f"naive score: {sum(Y_test) / 100}")
