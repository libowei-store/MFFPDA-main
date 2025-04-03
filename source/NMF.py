import numpy as np
from sklearn.decomposition import NMF


def NMF_calculate(M, k=2):
    nmf = NMF(n_components=k,
              init=None,
              solver='cd',
              beta_loss='frobenius',
              tol=1e-4,
              max_iter=200,
              random_state=None,
              l1_ratio=0.,
              verbose=0,
              shuffle=False
              )
    nmf.fit(M)
    W = nmf.fit_transform(M)
    H = nmf.components_
    H = np.array(H).T

    return H
