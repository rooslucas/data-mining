from sklearn.metrics import confusion_matrix
from symbol import for_stmt
import numpy as np
from functions import *
from sklearn.model_selection import train_test_split
credit_data = np.genfromtxt(
    'Assignment_1/pima.txt', delimiter=',', skip_header=True)

x = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y = np.reshape(y, (len(y), 1))
# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
tree = tree_grow(x, y, 20, 5, x.shape[1])
# print(RenderTree(tree))
# print(tree_pred(x, tree))
# trees = tree_grow_b(x=x, y=y, nmin=20, minleaf=5, nfeat=x.shape[1], m=5)
# print(trees)
# predictions = tree_pred_b(x, trees)
# true_labels = credit_data[:, credit_data.shape[1] - 1]
predictions = tree_pred(x, tree)
true_labels = credit_data[:, credit_data.shape[1] - 1]

tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
print(tn, fp, fn, tp)

specifity_1 = tn / (tn + fp)
print(f'Specifity naive 1: {specifity_1}')

sensitivity_1 = tp / (tp + fn)
print(f'Sensitivity naive 1: {sensitivity_1}')


PPV_1 = tp / (tp + fp)
NPV_1 = tn / (tn + fn)

print(PPV_1)
print(NPV_1)
print(sum(predictions == true_labels) / len(true_labels))
