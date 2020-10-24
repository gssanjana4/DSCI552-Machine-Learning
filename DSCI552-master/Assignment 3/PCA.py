"""
This Program is implemented as part of Assignment 3 for DSCI 552
Contributors: Sanjana Gopnal Swamy & Amit Sankhla
"""

import numpy as np

if __name__ == '__main__':
    data_file = "pca-data.txt"
    original_num_of_dims = 3
    target_num_of_dims = 2
    data = np.genfromtxt(data_file, delimiter='\t')
    mean = np.mean(data, axis=0)
    data_adj = (data - mean).T
    covariance = np.matmul(data_adj, data_adj.T) / len(data)
    eigen_value, eigen_vector = np.linalg.eig(covariance)
    principal_comp = [[value, eigen_vector[:, idx]] for idx, value in enumerate(eigen_value[: target_num_of_dims])]

    transformed_data = []
    for old_point in data:
        tmp = np.asarray([value[1] for value in principal_comp])
        new_point = tmp.dot(old_point).tolist()
        transformed_data.append(new_point)
    transformed_data = np.asarray(transformed_data)

    for idx, value in enumerate(principal_comp):
        print("Direction for Principal Component ", idx, " - ", value[1].tolist())
