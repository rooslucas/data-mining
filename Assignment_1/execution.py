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
tree = tree_grow(x, y, 2, 1, x.shape[1])
print(RenderTree(tree))
print(tree_pred(x, tree))
forest = RandomForest()
trees = forest.tree_grow_b(x, y, 15, 5, 41, m=5)
print(trees)
print(forest.tree_pred_b(trees, x))
# print(x)
# print(y)
