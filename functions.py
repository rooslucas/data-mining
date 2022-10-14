
# Assignment 1 ~ Classification Trees, Bagging and Random Forest
# Rosalie Lucas 6540384
# Michael Pieke 8474752
# Mick Richters 6545572

# Part 1: Programming

# Import necessary libraries
import pandas as pd
import random
from anytree import Node, RenderTree
from collections import Counter
import numpy as np

random.seed(12)
np.random.seed(42)
#################################################### MAIN FUNCTIONS ###########################################################


# Create a single DT
def tree_grow(x, y, nmin, minleaf, nfeat):
    # build_tree is used as auxiliary function in order for tree_grow to have the same paramaters as in the assignment
    root = Node("root", parent=None, question=Question(
        None, None, None))  # root node to be expanded
    return build_tree(root, x, y, nmin, minleaf, nfeat)

# Predict labels for x-values given a single decision tree


def tree_pred(x, tr):
    predicted_y = []  # Safe predictions here
    features = range(x.shape[1])  # Define features index

    # Predict for each x
    for row in x:
        # Get class from the tree
        predicted_y.append(get_decision(row, tr, features))

    # Return array with predictions
    return np.array(predicted_y)

# Grow a forest of decision trees


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    # Create forest of m trees
    forest = []  # Safe the trees in here
    tester_x = None
    #test = []
    for i in range(m):
        x, y = bootstrap(x, y, len(x))  # Get bootstrapped sample
        tree = tree_grow(x, y, nmin=nmin, minleaf=minleaf,
                         nfeat=nfeat)  # Grow a single tree based on bootstrapped sample
        pred = tree_pred(x, tree)
        # print(pred)
        if np.all(pred == 1):
            print('yes')
            # test.append(tree)
        if np.all(pred == 0):
            print('yes')
            # test.append(tree)
        forest.append(tree)

    return forest


# Make prediction for y values from the forest
def tree_pred_b(x, forest):
    # Create variables to safe the predictions
    tree_preds = np.zeros((0, len(x)))
    predictions = []

    # For each tree in the forest, make predictions for y values
    for tree in forest:
        tree_preds = np.append(tree_preds, [tree_pred(x, tree)], axis=0)

    # Transpose the predictions to get majority vote
    tree_preds = np.transpose(tree_preds)

    # Calculate majority vote over all trees for each x value (with binary data, majority vote == average value)
    for i in range(len(tree_preds)):
        majority_vote = Counter(tree_preds[i]).most_common(1)[0][0]
        # Final predictions of the random forest
        predictions.append(majority_vote)

    return predictions

################################################ AUXILIARY FUNCTIONS FOR TREE_GROW ###########################################################

# The real building


def build_tree(node, x, y, nmin, minleaf, nfeat):
    # Base Case: stop expanding if node is pure

    if nfeat < x.shape[1]:  # get random subset of features of size nfeat
        features = random.sample(range(x.shape[1]), nfeat)

    else:  # Use all features
        features = range(x.shape[1])

    split, feature, gain = best_split(x, y, features, minleaf)

    # if split == None, no minleaf criteria have been met
    if gain == 0 or len(x) < nmin or split == None:
        y_temp = np.reshape(y, (y.shape[0],))  # Reshape to fit counter
        majority_class = Counter(y_temp.tolist()).most_common(1)[0][0]
        #majority_class = y[0][0]
        # print(majority_class)
        # print()
        # Get final class value, should be the same for all y in Y as node is pure. Hence, we can select the first element of y.

        # does this make sense with question etc?

        # TODO: Not sure if this will duplicate the same node and can't access the parent nodes to change them
        leaf = Node("leaf " + str(majority_class), parent=node,
                    question=Question(None, None, None))
        leaf.question.label = majority_class
        #node.children = [leaf]
        #node.name = "leaf " + str(majority_class)
        return leaf

    # Return the best value to split on as well as the corresponding feature (column index)

    # Partition the data according to the best_split
    lhs_x, lhs_y, rhs_x, rhs_y, question = partition(feature, split, x, y)

    lhs_y = np.reshape(lhs_y, (lhs_y.shape[0], 1))  # Reshape to fit the format
    rhs_y = np.reshape(rhs_y, (rhs_y.shape[0], 1))  # Reshape to fit the format

    # Early Stopping Criteria (as provided in the assignment)
    '''if (len(x) < nmin):
        #print("early stopping")
        #print(y)
        y_temp = np.reshape(y, (y.shape[0],))  # Reshape to fit counter
        majority_class = Counter(y_temp.tolist()).most_common(1)[0][0]
        #print(majority_class)
        #print()

        # Create leaf node with majority class label
        question.label = majority_class
        leaf = Node("leaf " + str(majority_class), parent=node, question=question)
        node.children = [leaf]

        return leaf'''

    # Expand left and right side of tree using recursion
    node_lhs = build_tree(Node("lhs " + str(question.feature) + " value: " + str(np.round(question.value, 3)), parent=node, question=question),
                          lhs_x, lhs_y, nmin, minleaf, nfeat)
    node_rhs = build_tree(Node("rhs " + str(question.feature) + " value: " + str(np.round(question.value, 3)), parent=node, question=question),
                          rhs_x, rhs_y, nmin, minleaf, nfeat)

    node.children = [node_lhs, node_rhs]
    node.question = question

    # When finished building, return node
    return node


# Calculate impurity using Gini-Index
def impurity(y):
    if len(y) > 0:
        prob1 = sum(y) / len(y)
        prob2 = 1 - prob1
        return prob1*prob2

    else:
        return 0


# Calculate impurity reduction for a node
def impurity_reduction(y, lh, rh):
    propl = len(lh) / len(y)
    propr = len(rh) / len(y)
    imp = impurity(y)
    reduction = imp - ((propl * impurity(lh)) + (propr * impurity(rh)))

    return reduction

# Find the split which provides highest information gain


def best_split(x, y, features, minleaf):
    # Set default values
    best_gain = 0
    split_feature = random.choice(features)
    x_sorted = np.sort(np.unique(x[:, split_feature]))

    if len(x_sorted) > 1:
        best_split = random.choice((x_sorted[0:(len(x_sorted)-1)] +
                                    x_sorted[1:len(x_sorted)])/2)
    else:
        best_split = x_sorted[0]

    # Loop through all features to define the best feature to split on
    for feature in features:
        column = x[:, feature]
        x_sorted = np.sort(np.unique(column))

        # Define the splitpoints
        splitpoints = (x_sorted[0:(len(x_sorted)-1)] +
                       x_sorted[1:len(x_sorted)])/2
        # Calculate the impurity reduction for each split point
        for splitpoint in splitpoints:
            lh = y[column < splitpoint]
            rh = y[column >= splitpoint]
            if len(lh) == 0 or len(rh) == 0:
                continue

            gain = impurity_reduction(y, lh, rh)

            # Only update best gain if minleaf constraint is met
            if (gain > best_gain) and (len(lh) >= minleaf) and (len(rh) >= minleaf):
                best_gain = gain
                best_split = splitpoint
                split_feature = feature

    # Return the value and feature for the best split
    return best_split, split_feature, best_gain


# Partition based on given feature and split value
def partition(feature, split, x, y):
    question = Question(feature, split)
    # Easier to get correct labels for split data
    x = np.concatenate((x, y), axis=1)
    # Split data into left and right hand sides

    lhs = x[x[:, feature] < split]
    #np.array([instance for instance in x if instance[feature] < split])
    #rhs = np.array([instance for instance in x if instance[feature] >= split])
    rhs = x[x[:, feature] >= split]
    '''except:
        print(x[:,feature])
        print(split)
        print(x[:,feature] < split)'''
    columns = x.shape[1]-1

    # Return right and left side x and y values to use for next iteration of build_tree function
    return lhs[:, 0: columns], lhs[:, columns], rhs[:, 0: columns], rhs[:, columns], question


#################################################### AUXILIARY FUNCTIONS FOR TREE_PRED ###########################################################

# Get the final class for a leaf node
def get_decision(row, tr, features):
    decision = None

    # Loop through each feature to see which one matches the current node
    while True:
        # Check if root node is leaf node
        if type(tr) == list:
            print(tr)
        if tr.question.feature == None:
            return tr.question.label

        # Match feature with child feature
        # for child in tr.children:
         #   print(child.name)
        # print()
        for feat in features:
            if (tr.question.feature == feat):
                # If feature value >= value in question, get right hand of tree
                if (tr.question.answer(row)):
                    tr = tr.children[1]

                # Otherwise get left hand of tree
                else:
                    tr = tr.children[0]

            # Check for leaf node and return correct class
            if tr.name.startswith("leaf"):
                decision = tr.question.label
                return int(decision)

# Get child of a node


def get_child(node, index):
    # If node is a leaf, return node
    return node.children[index]
    # for child in node.children:
    # if (child.name == child_name):
    #    return child
    # if  (child.name == "leaf"):
    # return child


# Auxiliary class to store information based on which the data was split at each iteration of build_tree
class Question:

    def __init__(self, feature, value, label=None):
        # Question stores the feature, value and label of a node
        self.feature = feature
        self.value = value  # This value is the right hand side of the boolean, the left hand side is the feature value of the example
        self.label = label

    def answer(self, instance):
        val = instance[self.feature]

        # Return whether value is bigger than split value or not
        return val >= self.value


################################################ AUXILIARY FUNCTION FOR RANDOM FOREST ###########################################################


# Get a bootstrap sample from the data
def bootstrap(X, Y, n_bootstraps):
    bootstrap_indices = np.random.randint(
        low=0, high=len(X), size=n_bootstraps)

    df_bootstrapped_x = X[bootstrap_indices]
    df_bootstrapped_y = Y[bootstrap_indices]

    # Return the bootstrapped x and y values
    return df_bootstrapped_x, df_bootstrapped_y
