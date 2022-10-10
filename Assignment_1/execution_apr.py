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

# Load Data
credit_data = np.genfromtxt(
    'Assignment_1/credit.txt', delimiter=',', skip_header=True)
bug_data = np.genfromtxt(
    'Assignment_1/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', delimiter=';', skip_header=True)

x = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y = np.reshape(y, (len(y), 1))
target = []
for i in y:
    target.append(int(i))


# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
#tree = tree_grow(x, y, 2, 1, x.shape[1])
trees = tree_grow_b(x, y, 15, 5, 41, m=1)
# print(RenderTree(trees[3]))
# print(trees)
trees_p = tree_pred_b(x, trees)
#print(tree_pred_b(x, trees))
# print(x)
# print(y)

a = [1, 1, 1]
b = [1, 1, 1]


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


print(precision(a, b))
print(recall(a, b))
print(accuracy(target, trees_p))
