import numpy as np
import pandas as pd
def dual_transformation_networks_calculate(network):
    arr1 = []
    arr2 = []
    step1 = np.zeros((network.shape[0], network.shape[1]))
    step2 = np.zeros((network.shape[0], network.shape[0]))
    for i in range(network.shape[0]):
        temp = np.where(network[i] == 1)[0]
        arr1.append(temp)
    for i in range(len(arr1)):
        length = len(arr1[i])
        for j in range(length):
            step1[i][arr1[i][j]] = 1 / length
    step1 = step1.T

    network2 = network.T
    for i in range(len(network2)):
        temp = np.where(network2[i] == 1)[0]
        arr2.append(temp)
    for i in range(len(arr2)):
        length = len(arr2[i])
        for j in range(length):
            step2[arr2[i][j]] += step1[i] / length
    return step2

# data = pd.read_csv('../probiotic_data/association_matrix.csv', header=None, index_col=None)
# print(data.shape)
# net = data.to_numpy()
# result1 = dual_transformation_networks(net)
# result2 = dual_transformation_networks(net.T)
# print(result1.shape)
# print(result2.shape)

def dual_transformation_networks(network):
    result1 = dual_transformation_networks_calculate(network)
    result2 = dual_transformation_networks_calculate(network.T)

    return result1, result2
