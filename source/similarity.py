import numpy as np
import math
import scipy.spatial.distance as dist

def HIP_Calculate(M):
    l = len(M)
    cl = np.size(M, axis=1)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k] != M[j][k]:
                    dnum = dnum + 1
            SM[i][j] = 1 - dnum / cl  # HIP计算出来的相似矩阵
    return SM



def Cosine_similarity(M):
    l = len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1 = np.dot(M[i], M[j])
            v2 = np.linalg.norm(M[i], ord=2)
            v3 = np.linalg.norm(M[j], ord=2)
            if v2 * v3 == 0:
                SM[i][j] = 0
            else:
                SM[i][j] = v1 / (v2 * v3)
    return SM


# sigmoid function kernel similarity
def sigmoid_similarity(MD):
    m = MD.shape[0]
    sig_MS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i, :]
            b = MD[j, :]
            z = (1 / m) * (np.dot(a, b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)

    sig_MS1 = np.array(sig_MS1).reshape(m, m)
    return sig_MS1



def Jaccard_similarity(M):
    arr = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            arr[i, j] = dist.pdist(np.array([M[i,:], M[j,:]]), "jaccard")
    return arr