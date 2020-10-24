"""
This Program is implemented as part of Assignment 4 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import numpy as np


def train():
    n, d = X.shape
    weights = np.random.random(d)
    iteration = 0
    print("\n")
    while iteration < max_iterations:
        gradient = np.zeros(d)
        for x, y in zip(X, Y):
            s = y * np.dot(weights, x)
            tmp = 1 + np.exp(s)
            gradient = np.add(gradient, (x * y) / tmp)
        gradient /= n
        weights += alpha * gradient
        iteration += 1
        if iteration % 1000 == 0:
            print('{0} iterations completed'.format(iteration))

    print("----- Training Complete -----")
    return weights, iteration


def predict():
    prediction = [None for _ in range(X.shape[0])]
    for idx, x in enumerate(X):
        s = 1 * np.dot(weights, x)
        prob = np.exp(s) / (1 + np.exp(s))
        if prob > 0.5:
            prediction[idx] = 1
        else:
            prediction[idx] = -1

    return np.asarray(prediction)


if __name__ == '__main__':
    alpha = 0.01
    max_iterations = 7000
    data_file = "classification.txt"

    print("\nLogistic Regression")
    print("Max Iterations = ", max_iterations)
    print("Learning Rate = ", alpha)

    data = np.genfromtxt(data_file, delimiter=",", dtype="float", usecols=(0, 1, 2, 4))
    X = np.insert(data[:, :-1], 0, 1, axis=1)
    Y = data[:, -1]

    weights, iteration = train()
    prediction = predict()
    accuracy = np.where(Y == prediction)[0].shape[0] / prediction.shape[0]

    print("\nResult")
    print("No. of iteration = ", iteration)
    print("Weights = ", weights)
    print("Accuracy = ", accuracy)
