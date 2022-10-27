
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

rf = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=37, max_features='sqrt')
rf.fit(X_train, Y_train)
print(rf.score(X_train, Y_train))

lr = LogisticRegression(C=0.615848211066026, penalty='l1',
                        solver='liblinear', random_state=37)
lr.fit(X_train, Y_train)
print(lr.score(X_train, Y_train))

print(sum(Y_test))

dt = DecisionTreeClassifier
# dt.fit(X_train, Y_train)
# print(dt.score(X_dev, Y_dev))
