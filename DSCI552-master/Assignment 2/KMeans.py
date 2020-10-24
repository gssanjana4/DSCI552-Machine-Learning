"""
This Program is implemented as part of Assignment 2 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import math
import random
import numpy


def kmeans(k, max_iterations):
    prev_cluster = []
    cur_cluster = random.sample(data, k)

    iteration = 0
    while sorted(prev_cluster) != sorted(cur_cluster) and iteration < max_iterations:
        for i in range(len(data)):
            distance = math.inf
            near_centroid = None
            for j in range(len(cur_cluster)):
                d = math.sqrt((data[i][0] - cur_cluster[j][0]) ** 2 + (data[i][1] - cur_cluster[j][1]) ** 2)
                if d < distance:
                    distance = d
                    near_centroid = j
            data[i][2] = near_centroid

        prev_cluster = cur_cluster
        grouped_list = [[[y[0], y[1]] for y in data if y[2] == x] for x in range(k)]
        cur_cluster = [numpy.mean(each_row, axis=0).tolist() for each_row in grouped_list]

        iteration += 1

    return cur_cluster


if __name__ == '__main__':
    data = []
    num_of_centroids = 3
    max_num_of_iterations = 200
    file = open("clusters.txt", "r")
    for line in file:
        temp = list(map(float, line.strip().split(",")))
        temp.append(0)
        data.append(temp)

    print(data)
    centroids = kmeans(num_of_centroids, max_num_of_iterations)
    print([[round(item[0], 3), round(item[1], 3)] for item in centroids])

