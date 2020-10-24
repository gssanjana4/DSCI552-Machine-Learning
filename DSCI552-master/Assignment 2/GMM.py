"""
This Program is implemented as part of Assignment 2 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import numpy as np
from collections import defaultdict
import random


def GMM():
    membership_weights = None
    iteration = 0
    while iteration < max_num_of_iterations:
        mstep(membership_weights)
        new_membership_weights = estep()
        if membership_weights is not None and new_membership_weights is not None:
            if (np.abs(new_membership_weights - membership_weights) < threshold).all():
                break

        membership_weights = new_membership_weights
        iteration += 1


def mstep(membership_weights):
    global means, cov_vars, amplitude
    means = []
    cov_vars = []
    amplitude = []
    num_of_var = len(data[0])
    num_of_data_points = len(data)

    if membership_weights is not None:
        amplitude = np.sum(membership_weights, axis=0) / num_of_data_points
        for i in range(0, num_of_clusters):
            means.append(np.sum(np.multiply(data, membership_weights[:, i].reshape(len(data), 1)), axis=0) / np.sum(
                membership_weights[:, i]))
            cov_temp_sum = np.zeros((num_of_var, num_of_var))
            for j in range(0, num_of_data_points):
                temp = data[j] - means[i]
                temp = np.dot(temp.T.reshape(num_of_var, 1), temp.reshape(1, num_of_var))
                temp = temp * membership_weights[j][i]
                cov_temp_sum = np.add(cov_temp_sum, temp)
            cov_temp_sum = cov_temp_sum / sum(membership_weights[:, i])
            cov_vars.append(cov_temp_sum)
    else:
        clusters = kmeans()
        #print(clusters)
        amplitude = np.ones(num_of_clusters) / num_of_clusters
        amplitude = amplitude.tolist()
        for i in range(0, num_of_clusters):
            means.append(np.mean(clusters[i], axis=0))
            cov_vars.append(np.cov(clusters[i].T))


def kmeans():
    cluster_centroids = np.array(random.sample(list(data), 3))
    dict_of_clusters = defaultdict(list)
    for i in range(0, len(data)):
        val, min_index = min(
            (val, idx) for (idx, val) in enumerate([euclidean_dist(data[i], x) for x in cluster_centroids]))
        dict_of_clusters[min_index].append(data[i].tolist())
    dict_of_clusters = [np.array(dict_of_clusters[i]) for i in dict_of_clusters]
    return dict_of_clusters


def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def estep():
    num_of_data_points = len(data)
    pdfs = np.empty([num_of_data_points, num_of_clusters])
    for i in range(0, num_of_clusters):
        m = means[i]
        cov = cov_vars[i]
        invcov = np.linalg.inv(cov)
        norm_factor = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))
        for row in range(0, num_of_data_points):
            temp = data[row, :] - m
            temp = temp.T
            temp = np.dot(-0.5 * temp, invcov)
            temp = np.dot(temp, (data[row, :] - m))
            pdfs[row][i] = norm_factor * np.exp(temp)
    membership_weights = np.empty([num_of_data_points, num_of_clusters])
    for i in range(0, num_of_data_points):
        denominator = np.sum(amplitude * pdfs[i])
        for j in range(0, num_of_clusters):
            membership_weights[i][j] = amplitude[j] * pdfs[i][j] / denominator
    return membership_weights


if __name__ == '__main__':
    num_of_clusters = 3
    max_num_of_iterations = 200
    threshold = 0.01
    means = cov_vars = amplitude = []
    data = np.genfromtxt("clusters.txt", delimiter=',')
    GMM()
    print("Amplitudes:")
    print(np.round(amplitude, 3).tolist())
    print("Means:")
    print(np.round(np.array(means), 3).tolist())
    print("Covariance Matrix:")
    print(np.round(np.array(cov_vars), 3).tolist())

