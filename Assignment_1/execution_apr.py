# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:06:23 2022

@author: Mick_
"""

from symbol import for_stmt
import numpy as np
from functions import *
from sklearn.model_selection import train_test_split
credit_data = np.genfromtxt(
    'C:/Users/Mick_/OneDrive/Documenten/Data_mining/assignment 1/data-mining-master/Assignment_1/credit.txt', delimiter=',', skip_header=True)
bug_data = np.genfromtxt(
    'C:/Users/Mick_/OneDrive/Documenten/Data_mining/assignment 1/data-mining-master/Assignment_1/eclipse-metrics-packages-2.0.txt', delimiter=',', skip_header=True)

x = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y = np.reshape(y, (len(y), 1))
target = []
for i in y:
    target.append(int(i))
    

# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
#tree = tree_grow(x, y, 2, 1, x.shape[1])
trees = tree_grow_b(x, y, 15, 5, 41, m=1)
#print(RenderTree(trees[3]))
#print(trees)
trees_p = tree_pred_b(x, trees)
#print(tree_pred_b(x, trees))
# print(x)
# print(y)

a = [1, 1, 1]
b = [1, 1, 1]

def accuracy (y_true, y_pred):
    acc = np.sum(np.equal(y_true, y_pred)) / len(y_true)
    return acc
def precision (y_true, y_pred):
    TP = 0
    FP = 0
    for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] ==  1:
                TP += 1
            if  y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
    prec = TP/ (TP + FP)
    return prec

def recall (y_true, y_pred):
    TP = 0
    FN = 0
    for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] ==  1:
                TP += 1
            if  y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
    rec = TP/ (TP + FN)
    return rec
print(precision(a,b))
print(recall(a,b))
print(accuracy(target, trees_p))