# Assignment 1 ~ Classification Trees, Bagging and Random Forest

# Rosalie Lucas 6540384
# Michael Pieke 8474752
# Mick Richters 6545572

# Part 2: Analysis

# Import libraries
from symbol import for_stmt
import numpy as np
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load Data
credit_data = np.genfromtxt(
    'Assignment_1/pima.txt', delimiter=',', skip_header=True)
bug_data = np.genfromtxt(
    'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)

x_train = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y_train = np.reshape(y, (len(y), 1))

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42)


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


# x_train, y_train = slice_data(pd.read_csv(
#     'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', delimiter=';'))
# x_test, y_test = slice_data(pd.read_csv(
#     'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv', delimiter=';'))

target = []
for i in y_train:
    target.append(int(i))

# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
tree = tree_grow(x_train, y_train, 15, 5, 9)
trees = tree_grow_b(x_train, y_train, 15, 5, 9, m=100)
forest = tree_grow_b(x_train, y_train, 15, 5, 3, 100)
# print(RenderTree(trees[3]))
# print(trees)

tree_p = tree_pred(x_train, tree)
trees_p = tree_pred_b(x_train, trees)
forest_p = tree_pred_b(x_train, forest)
#print(tree_pred_b(x, trees))
# print(x)
# print(y)


def accuracy(y_true, y_pred):
    acc = np.sum(np.equal(y_true, y_pred)) / len(y_true)
    return acc


def precision(y_true, y_pred):
    TP = 0
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
    prec = TP / (TP + FP)
    return prec


def recall(y_true, y_pred):
    TP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    rec = TP / (TP + FN)
    return rec


# print(precision(a, b))
# # print(recall(a, b))
print(accuracy(target, tree_p))
tn, fp, fn, tp = confusion_matrix(target, tree_p).ravel()
print(tn, fp, fn, tp)

print(accuracy(target, trees_p))
tn, fp, fn, tp = confusion_matrix(target, trees_p).ravel()
print(tn, fp, fn, tp)

print(accuracy(target, forest_p))
tn, fp, fn, tp = confusion_matrix(target, forest_p).ravel()
print(tn, fp, fn, tp)
