# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:51:44 2022

@author: scro0
"""

#Part 2: analysis

import pandas as pd
import numpy as np
import seaborn as sns
from functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import time

#Get the relevant features from bug dataset
def slice_data(data):
  x = data.loc[:, 'ACD_avg':'VG_sum']
  pre = data.loc[:,'pre']
  x = np.array(pd.concat([pre, x], axis = 1))
  y = np.array(data.loc[:,'post'])
  
  for i in range(len(y)): #binarise labels
      if y[i] > 1:
          y[i] = 1
         
  y = np.reshape(y, (y.shape[0], 1))

  return x, y

x_train, y_train = slice_data(pd.read_csv('eclipse-metrics-packages-2.0.csv', delimiter = ';'))
x_test, y_test = slice_data(pd.read_csv('eclipse-metrics-packages-3.0.csv', delimiter = ';'))

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

#Model 1

time_1 = time.perf_counter()
model_1 = tree_grow(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 41)
y_pred_1 = tree_pred(x_test, model_1)
time_2 = time.perf_counter()
results(y_test, y_pred_1, time_1, time_2, "single tree")

#Model 2
time_1 = time.perf_counter()
model_2 = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 41, m = 100)
y_pred_2 = tree_pred_b(x_test, model_2)
time_2 = time.perf_counter()
results(y_test, y_pred_2, time_1, time_2, "bagging model")

#Model 3
time_1 = time.perf_counter()
model_3 = tree_grow_b(x_train, y_train, nmin = 15, minleaf = 5, nfeat = 6, m = 100)
y_pred_3 = tree_pred_b(x_test, model_3)
time_2 = time.perf_counter()
results(y_test, y_pred_3, time_1, time_2, "random forest")

