# Assignment 1 ~ Classification Trees, Bagging and Random Forest
# Rosalie Lucas 6540384
# Michael Pieke
# Mick Richters

# Part 1: Programming

from os import major
import pandas as pd
import random
from anytree import Node, RenderTree
from collections import Counter
import numpy as np


def tree_grow(x, y, nmin, minleaf, nfeat):
    # separate function so that the parameters of this function stay the same
    return build_tree(Node("root", parent=None, question=Question(None, None, None)), x, y, nmin, minleaf, nfeat)


def build_tree(node, x, y, nmin, minleaf, nfeat):

    # base case: stop if node is pure
    if impurity(y) == 0:
        # get final class value, should be the same for all y in Y as node is pure
        node.question.label = y[0][0]
        # not sure if this will duplicate the same node and can't access the parent nodes to change them
        leaf = Node("leaf", parent=node, question=node.question)
        node.children = [leaf]
        return node

    if nfeat < x.shape[1]:
        # split is determined based on nfeat number of features
        features = random.sample(range(x.shape[1]), nfeat)

    else:
        features = range(0, x.shape[1]-1)  # -1 right?

    split, feature = best_split(x, y, features)
    # split into left and right hand sides and then build tree for both
    lhs_x, lhs_y, rhs_x, rhs_y, question = partition(feature, split, x, y)

    lhs_y = np.reshape(lhs_y, (lhs_y.shape[0], 1))
    rhs_y = np.reshape(rhs_y, (rhs_y.shape[0], 1))

    # early stopping criteria

    # make sure len corresponds to number of instances
    if (len(x) < nmin) or (len(lhs_x) < minleaf) or (len(rhs_x) < minleaf):
        # doesnt work with Counter otherwise
        y_temp = np.reshape(y, (y.shape[0],))
        majority_class = Counter(y_temp.tolist()).most_common(1)[0][0]
        question.label = majority_class
        leaf = Node("leaf", parent=node, question=question)
        node.children = [leaf]
        node.question = question
        return node

    node_lhs = build_tree(Node("lhs", parent=node, question=question),
                          lhs_x, lhs_y, nmin, minleaf, nfeat)  # expand left side of tree
    node_rhs = build_tree(Node("rhs", parent=node, question=question),
                          rhs_x, rhs_y, nmin, minleaf, nfeat)  # expand right side of tree

    node.children = [node_lhs, node_rhs]
    node.question = question

    return node


def impurity(y):  # gini-index
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


def best_split(x, y, features):
    # loop through each feature and find the feature which provides highest information gain
    best_gain = 0
    split_feature = None
    best_split = None

    # need to make compatible for categorical attributes
    for feature in features:
        column = x[:, feature]
        x_sorted = np.sort(np.unique(column))
        splitpoints = (x_sorted[0:(len(x_sorted)-1)] +
                       x_sorted[1:len(x_sorted)])/2
        for splitpoint in splitpoints:
            lh = y[column < splitpoint]
            rh = y[column >= splitpoint]
            gain = impurity_reduction(y, lh, rh)
            if gain > best_gain:
                best_gain = gain
                best_split = splitpoint
                split_feature = feature

    return best_split, split_feature


def partition(feature, split, x, y):  # partition based on feature. How to ask the question?

    question = Question(feature, split)
    x = np.concatenate((x, y), axis=1)  # easier to get labels for rhs and lhs
    # given numeric values, need to add for str values as well
    lhs = np.array([instance for instance in x if instance[feature] < split])
    rhs = np.array([instance for instance in x if instance[feature] >= split])
    columns = x.shape[1]-1

    return lhs[:, 0: columns], lhs[:, columns], rhs[:, 0: columns], rhs[:, columns], question

# https://www.youtube.com/watch?v=LDRbO9a6XPU


def get_child(node, child_name):

    for child in node.children:
        if (child.name == child_name) or (child.name == "leaf"):
            return child


def get_decision(row, tr, features):
    # need to loop through each feature to see which one matches the current node
    decision = None

    while True:
        for feat in features:  # match feature with child feature
            if (tr.question.feature == feat):
                # if feature value >= value in question, get rhs and lhs otherwise
                if (tr.question.answer(row)):
                    tr = get_child(tr, "rhs")
                else:
                    tr = get_child(tr, "lhs")

            if tr.name == "leaf":
                decision = tr.question.label
                return int(decision)


class Question:  # inspired by this tutorial: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

    def __init__(self, feature, value, label=None):
        # I want to make it so that we can match features of both examples and the question
        self.feature = feature
        self.value = value  # this value is the right hand side of the boolean, the left hand side is the feature value of the example
        self.label = label

    def answer(self, instance):
        val = instance[self.feature]
        # don't need to worry about non-numerical values, so this is fine
        return val >= self.value

# TODO: Write tree_grow(x, y, nmin, minleaf, nfeat) function


def tree_grow(x, y, nmin, minleaf, nfeat):
    # separate function so that the parameters of this function stay the same
    return build_tree(Node("root", parent=None, question=None), x, y, nmin, minleaf, nfeat)


# TODO: Write tree_pred function

def tree_pred(x, tr):
    y = []
    features = range(x.shape[1])
    for row in x:
        y.append(get_decision(row, tr, features))
    return np.array(y)

# Bagging
# TODO: Write tree_grow_b(x, y, nmin, minleaf, nfeat, m) function


def bootstrap(X, Y, n_bootstraps):
    bootstrap_indices = np.random.randint(
        low=0, high=len(X), size=n_bootstraps)
    df_bootstrapped_x = X[bootstrap_indices]
    df_bootstrapped_y = Y[bootstrap_indices]

    return df_bootstrapped_x, df_bootstrapped_y


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:

    # def __init__(self):
    #     self.trees = []

    def tree_grow_b(self, x, y, nmin, minleaf, nfeat, m):
        trees = []
        for i in range(m):
            x, y = bootstrap(x, y, len(x))
            tree = tree_grow(x, y,
                             nmin=nmin, minleaf=minleaf, nfeat=nfeat)
            trees.append(tree)

        return trees

        # bootstrap

        # for loop (for tree in n_trees) trekken bootstrap sample en daarmee boom trainen en de getrainde boom opslaan in self.trees

# TODO: Write tree_pred_b(m, x) function

    def tree_pred_b(self, trees, x):
        tree_preds = np.empty((0, len(x)))
        predictions = []

        for tree in trees:
            tree_preds = np.append(
                tree_preds, [tree_pred(x, tree)], axis=0)

        for i in range(len(tree_preds)):
            tree_preds = np.transpose(tree_preds)
            majority_vote = Counter(tree_preds[i]).most_common(1)[0][0]
            predictions.append(majority_vote)

        return predictions
