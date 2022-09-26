import numpy as np
import pandas as pd
import os
import random
#from decision_tree import DecisionTree
from collections import Counter
credit_data_txt = np.genfromtxt('C:/Users/Mick_/OneDrive/Documenten/Data_mining/credit.txt', delimiter=',', skip_header=True)
credit_data = pd.read_csv('C:/Users/Mick_/OneDrive/Documenten/Data_mining/credit.csv', delimiter=',')

inputs = credit_data_txt[:, range(0, 4)]
target = credit_data_txt[range(0, 10),5]

# def bootstrap(X, y):
#     n_samples = X.shape[0] #first dimension = #samples, second dimension = #features
#     indices = np.random.choice(n_samples, size = n_samples, replace = True)
   
def bootstrap(train_df, n_bootstraps): 
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstraps)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped
   
    
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    
    def __init__(self, n_trees=100, nmin = 15, minleaf=5, nfeat=41):
        self.n_trees = n_trees
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat
        self.trees = []
        
    def tree_grow_b(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(nmin=self.nmin, minleaf=self.minleaf, nfeat=self.minleaf)
            
        #bootstrap
       
        #for loop (for tree in n_trees) trekken bootstrap sample en daarmee boom trainen en de getrainde boom opslaan in self.trees

    def tree_pred_b(self, x):
        tree_preds = np.array([])
        #[1111 0000 1111] array of array's of predictions per tree
        #[101 101 101 101] convert to corresponding predictions per tree
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        #101 101 101 --> 111
        y_preds = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_preds)
        