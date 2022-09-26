import numpy as np
from functions import *
credit_data = np.genfromtxt(
    '/Users/roos/Developer/data-mining/Assignment_1/credit.txt', delimiter=',', skip_header=True)

x = credit_data[:, 0: credit_data.shape[1] - 1]
y = credit_data[:, credit_data.shape[1] - 1]
y = np.reshape(y, (10, 1))
# checked to see if the same as slides. It is the same, only leaf nodes are duplicated (i.e. parent of leaf == leaf). Probably good to fix.
tree = tree_grow(x, y, 2, 1, x.shape[1])
print(RenderTree(tree))
