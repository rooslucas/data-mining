# Assignment 1 ~ Classification Trees, Bagging and Random Forest
# Rosalie Lucas 6540384
# Michael Pieke
# Mick Richters

# Part 1: Programming

import numpy as np
credit_data = np.genfromtxt(
    'credit.txt', delimiter=',', skip_header=True)

print(credit_data)
