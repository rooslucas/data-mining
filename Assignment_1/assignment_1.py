# Assignment 1 ~ Classification Trees, Bagging and Random Forest
# Rosalie Lucas 6540384
# Michael Pieke
# Mick Richters

# Part 1: Programming

import numpy as np
credit_data = np.genfromtxt(
    '/Users/roos/Developer/data-mining/Assignment_1/credit.txt', delimiter=',', skip_header=True)

print(credit_data)

# TODO: Write tree_grow(x, y, nmin, minleaf, nfeat) function


def impurity(y):
    if len(y) > 0:
        prob1 = sum(y) / len(y)
        prob2 = 1 - prob1
        return prob1*prob2
    else:
        return 0


def impurity_reduction(y, lh, rh):
    propl = len(lh) / len(y)
    propr = len(rh) / len(y)
    imp = impurity(y)
    reduction = imp - ((propl * impurity(lh)) + (propr * impurity(rh)))
    return reduction


def best_split(x, y):
    x_sorted = np.sort(np.unique(x))
    splitpoints = (x_sorted[0:(len(x_sorted)-1)] + x_sorted[1:len(x_sorted)])/2

    # loop through each feature and find the feature which provides highest information gain
    best_gain = 0
    for instance in splitpoints:
        # for instance in x.iloc[i, :]:
        lh = y[x < instance]
        rh = y[x > instance]
        gain = impurity_reduction(y, lh, rh)
        if gain > best_gain:
            best_gain = gain
            best_split = instance
    return best_split


# TODO: Write tree_pred function

# Bagging
# TODO: Write tree_grow_b(x, y, nmin, minleaf, nfeat, m) function
# TODO: Write tree_pred_b(m, x) function

print(impurity(credit_data[:, 5]))
print(best_split(credit_data[:, 3], credit_data[:, 5]))
