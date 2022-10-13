# Assignment 1 ~ Classification Trees, Bagging and Random Forest

# Rosalie Lucas 6540384
# Michael Pieke 8474752
# Mick Richters 6545572

# Part 2: analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
from functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time
from anytree.exporter import DotExporter

# Get the relevant features from bug dataset


def slice_data(data):
    x = data.loc[:, 'ACD_avg':'VG_sum']
    pre = data.loc[:, 'pre']
    x = np.array(pd.concat([pre, x], axis=1))
    y = np.array(data.loc[:, 'post'])

    for i in range(len(y)):  # binarise labels
        if y[i] > 1:
            y[i] = 1

    y = np.reshape(y, (y.shape[0], 1))

    return x, y


x_train, y_train = slice_data(pd.read_csv(
    'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', delimiter=';'))
x_test, y_test = slice_data(pd.read_csv(
    'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv', delimiter=';'))

'''
Train a single classification tree on the training set with nmin = 15,
minleaf = 5 (we have pre-selected reasonable values for you), and nfeat
= 41. Compute the accuracy, precision and recall on the test set.

'''


def results(y_test, y_pred, time_1, time_2, model_name):
    print(f'Time elapsed for {model_name}: {abs(time_1 - time_2)} seconds')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    cf = confusion_matrix(y_test, y_pred)
    print(cf)
    print()

    sns.heatmap(cf, annot=True).set(title="Confusion matrix of " + model_name)


# Model 1
time_1 = time.perf_counter()
model_1 = tree_grow(x_train, y_train, nmin=15, minleaf=5, nfeat=41)
y_pred_1 = tree_pred(x_test, model_1)
time_2 = time.perf_counter()
results(y_test, y_pred_1, time_1, time_2, "single tree")

# Model 2
time_1 = time.perf_counter()
model_2 = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
y_pred_2 = tree_pred_b(x_test, model_2)
time_2 = time.perf_counter()
results(y_test, y_pred_2, time_1, time_2, "bagging model")

# Model 3
time_1 = time.perf_counter()
model_3 = tree_grow_b(x_train, y_train, nmin=15, minleaf=5, nfeat=6, m=100)
y_pred_3 = tree_pred_b(x_test, model_3)
time_2 = time.perf_counter()
results(y_test, y_pred_3, time_1, time_2, "random forest")

# Safe results
y_test = np.reshape(y_test, (1, y_test.shape[0]))
total_df = pd.DataFrame()
total_df['true_label'] = y_test[0]
total_df['pred_tree'] = y_pred_1
total_df['pred_bag'] = y_pred_2
total_df['pred_forest'] = y_pred_3
total_df.to_csv('results.csv', index=False)


# credit_data = np.genfromtxt(
#     "Assignment_1/credit.txt", delimiter=',', skip_header=True)
# x = credit_data[:, 0:credit_data.shape[1]-1]
# y = credit_data[:, credit_data.shape[1]-1]
# y = np.reshape(y, (len(y), 1))
# model_test = tree_grow(x, y, nmin=0, minleaf=0, nfeat=41)
# y_pred_test = tree_pred(x, model_test)
# y_pred_test = np.reshape(y_pred_test, (len(y_pred_test), 1))
# false_preds = np.where(y_pred_test != y)
# print(false_preds)
# false_labels = y[false_preds[0]]
# false_preds = x[false_preds[0]]

# #results(y_train, y_pred_1, time_1, time_2, "single tree")


# # This is to test if the DT code works for pima dataset


# pima = np.genfromtxt('Assignment_1/pima.txt', delimiter=',')

# x = pima[:, 0:pima.shape[1]-1]
# y = pima[:, pima.shape[1]-1]
# y = np.reshape(y, (768, 1))
# time_1 = time.perf_counter()
# model_1 = tree_grow(x, y, nmin=20, minleaf=5, nfeat=8)
# y_pred_1 = tree_pred(x, model_1)
# time_2 = time.perf_counter()
# results(y, y_pred_1, time_1, time_2, "RF")


# dt = DecisionTreeClassifier(
#     criterion='gini', min_samples_split=15, min_samples_leaf=5)

# dt_pima = dt.fit(x_train, y_train).predict(x_test)
# print(confusion_matrix(y_test, dt_pima))
# print(f'Accuracy: {accuracy_score(y_test, dt_pima)}')

# print(confusion_matrix(y_test, y_pred_1))
# print(f'Accuracy: {accuracy_score(y_test, y_pred_1)}')


# rf = RandomForestClassifier(n_estimators=100, criterion='gini',
#                             min_samples_split=15, min_samples_leaf=5, max_features=41)
# print(
#     f'Accuracy: {accuracy_score(y_test, rf.fit(x_train, y_train).predict(x_test))}')
# print(confusion_matrix(y_test, rf.fit(x_train, y_train).predict(x_test)))
# print(confusion_matrix(y_test, y_pred_3))

# # for random forest, some trees seem to always output 1, which heavily makes the predictions 1
