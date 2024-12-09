# -*- coding: utf-8 -*-
# @Time : 2024/4/25 上午9:59
# @Author : BoWei Li
# @File : probiotic_function_similarity.py
# @Software : PyCharm

import pandas as pd
import numpy as np

def probiotic_function_similarity_calculate(association, disease_semantic, probiotic_index1, probiotic_index2):
    list1 = association[probiotic_index1][:]
    list2 = association[probiotic_index2][:]

    index1 = np.where(list1 == 1)[0]
    index2 = np.where(list2 == 1)[0]
    arr1, arr2 = [], []
    for i in index1:
        temp = []
        for j in index2:
            temp.append(disease_semantic[i][j])
        if len(temp) != 0:
            Max = max(temp)
            arr1.append(Max)
    for i in index2:
        temp = []
        for j in index1:
            temp.append(disease_semantic[i][j])
        if len(temp) != 0:
            Max = max(temp)
            arr2.append(Max)
    if len(arr1) == 0 and len(arr2) == 0:
        similarity_score = 0
    else:
        similarity_score = (sum(arr1) + sum(arr2)) / (len(arr1) + len(arr2))
    return similarity_score



def probiotic_function_similarity(association_matrix, disease_semantic_matrix):
    length = association_matrix.shape[0]
    probiotic_function_matrix = np.eye(length)
    for i in range(length):
        for j in range(length):
            probiotic_function_matrix[i][j] = probiotic_function_similarity_calculate(association_matrix, disease_semantic_matrix, i, j)
    return probiotic_function_matrix
