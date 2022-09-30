from symbol import for_stmt
import numpy as np
from functions import *
from sklearn.model_selection import train_test_split
credit_data = np.genfromtxt(
    'Assignment_1/credit.txt', delimiter=',', skip_header=True)

x = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y = np.reshape(y, (len(y), 1))
# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
tree = tree_grow(x, y, 2, 1, 8)
print(RenderTree(tree))
print(tree_pred(x, tree))
trees = tree_grow_b(x=x, y=y, nmin=2, minleaf=3, nfeat=8, m=5)
print(trees)
print(tree_pred_b(trees, x))
# print(x)
# print(y)
