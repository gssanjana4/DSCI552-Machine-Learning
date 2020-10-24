"""
This Program is implemented as part of Assignment 4 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    alpha = 0.01
    max_iterations = 7000
    data_file = "classification.txt"

    print("--------- Pocket Algorithm ---------")
    print("Max Iterations = ", max_iterations)
    print("Learning Rate = ", alpha)

    data = np.genfromtxt(data_file, delimiter=",", dtype="float", usecols=(0, 1, 2, 4))
    X = np.insert(data[:, :-1], 0, 1, axis=1)
    Y = data[:, -1]

    weights = np.random.random(X.shape[1])
    best_weights = []
    print("Initial Weights = ", weights, end="\n\n")

    iteration = 0
    best_iteration = -1
    error_counts = []
    best_error_count = float("inf")

    while iteration < max_iterations:
        error_counts.append(0)
        for x, y in zip(X, Y):
            prod = np.dot(x, weights)
            if prod > 0 and y < 0:
                weights -= alpha * x
            elif prod < 0 and y > 0:
                weights += alpha * x

        tmp = np.sign(np.dot(X, weights))
        error_counts[-1] = X.shape[0] - np.where(tmp == Y)[0].shape[0]
        if best_error_count > error_counts[-1]:
            best_error_count = error_counts[-1]
            best_iteration = iteration
            best_weights = copy.deepcopy(weights)
        iteration += 1
        if iteration % 100 == 0:
            print('\r----- Training in progress : {0} iterations completed -----'.format(iteration), end="")
        if error_counts[-1] == 0:
            break

    print("\r----- Training Complete -----")

    prediction = np.sign(np.dot(X, weights))
    accuracy = np.where(Y == prediction)[0].shape[0] / prediction.shape[0]
    print("\n----- Result after final iteration (Perceptron) -----")
    print("No. of iteration = ", iteration)
    print("Weights = ", weights)
    print("Accuracy = ", accuracy)

    best_prediction = np.sign(np.dot(X, best_weights))
    best_accuracy = np.where(Y == best_prediction)[0].shape[0] / best_prediction.shape[0]
    print("\n----- Best Result (Pocket) -----")
    print("Best iteration = ", best_iteration)
    print("Best weights = ", best_weights)
    print("Best Accuracy / Least mismatches = {0} / {1}".format(best_accuracy, best_error_count))

    plt.ylabel('No. of violations')
    plt.xlabel('iteration')
    plt.plot(error_counts)
    plt.show()
