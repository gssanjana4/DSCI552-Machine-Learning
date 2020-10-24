"""
This Program is implemented as part of Assignment 3 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import random
import numpy
import matplotlib.pyplot as plt


def distance(a, b):
    if a == b:
        return 0
    return data[min(a, b), max(a, b)]


def calculate_farthest_points():
    cur_point = random.randint(1, numPoints)
    next_point = -1
    prev_point = -1
    max_dist = -1
    while True:
        for point in range(1, numPoints + 1):
            dist = distance(cur_point, point)
            if dist > max_dist:
                max_dist = dist
                next_point = point

        if next_point == prev_point:
            break

        cur_point, prev_point = next_point, cur_point
        max_dist = -1

    return min(cur_point, prev_point), max(cur_point, prev_point)


def recompute_distance(dim):
    res = dict()
    for i in range(1, numPoints + 1):
        for j in range(i + 1, numPoints + 1):
            res[i, j] = (data[i, j] ** 2 - (dim[i - 1] - dim[j - 1]) ** 2) ** 0.5

    return res


def fastmap():
    global data
    res = [[] for pt in range(numPoints)]
    for dim in range(numDims):
        pt1, pt2 = calculate_farthest_points()
        for point in range(1, numPoints + 1):
            if point == pt1:
                dist = 0
            elif point == pt2:
                dist = distance(pt1, pt2)
            else:
                dist = (distance(pt1, point) ** 2 + distance(pt1, pt2) ** 2 - distance(pt2, point) ** 2) / (
                            2 * distance(pt1, pt2))

            res[point - 1].append(dist)

        last_dim = list(map(lambda x: x[-1], res))
        data = recompute_distance(last_dim)

    return res


def plot(res):
    np_result = numpy.asarray(res)
    fig, ax = plt.subplots()
    ax.scatter(np_result[:, 0], np_result[:, 1])
    for idx, label in enumerate(word_list):
        ax.annotate(label, (np_result[idx]))

    plt.show()


if __name__ == '__main__':
    dataFile = "fastmap-data.txt"
    wordFile = "fastmap-wordlist.txt"
    numDims = 2

    word_list = []

    file = open(wordFile, "r")
    for line in file:
        word_list.append(line.strip())

    numPoints = len(word_list)
    data = dict()

    file = open(dataFile, "r")
    for line in file:
        p1, p2, d = line.split()
        data[int(p1), int(p2)] = float(d)

    result = fastmap()
    print(result)
    plot(result)
