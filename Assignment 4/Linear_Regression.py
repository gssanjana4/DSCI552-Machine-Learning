"""
This Program is implemented as part of Assignment 4 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import numpy as np
from numpy.linalg import inv

if __name__ == '__main__':
    data_file = "linear-regression.txt"
    print("\nLinear Regression Result")
    data = np.genfromtxt(data_file, delimiter=",", dtype="float", usecols=(0, 1, 2))
    X = np.insert(data[:, :-1], 0, 1, axis=1)
    Y = data[:, -1]
    weights = inv(X.T.dot(X)).dot(X.T).dot(Y)
    print("\nWeights = ", weights)
