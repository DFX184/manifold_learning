from scipy.linalg import eigh
from utils import *
import numpy as np

import numpy as np


def get_weights(data, nbors_idx, reg_func=None):
    """
    Calculate weights

    Parameters
    ----------
    data: (d,n) array, Input data
        d is its dimensionality
        n is the number of points. 
    nbors: (n,k) array. Indices of neghbours
        n is the number of points 
        k is the number of neighbours
    reg: regularization function

    Returns
    -------
    weights: (n,n) array. Weight matrix in row-major order
        weights[i,:] is weights of x_i
    """

    n = data.shape[1]
    weights = np.zeros((n, n))

    eps = 1e-3
    for i in range(n):
        x = data[:, i].reshape(-1, 1)
        k = nbors_idx[i].shape[0]  # number of neighbors
        ones = np.ones((k, 1))

        # k-neareast neighbors
        eta = data[:, nbors_idx[i]]
        eta_t = eta.T
        C = eta_t.dot(eta)

        # regularization term
        if reg_func is None:
            trace = np.trace(C)
            if trace > 0:
                R = eps/k*trace
            else:
                R = eps
            C += np.eye(k)*R
        else:
            C += reg_func(C, k)

        # C_inv = np.linalg.inv(C)
        C_inv = np.linalg.pinv(C)

        # calculate lagranian multipler lamda
        tmp = eta_t.dot(x)
        lam_num = 1. - ones.T.dot(C_inv).dot(tmp)
        lam_denom = ones.T.dot(C_inv).dot(ones)
        lam = lam_num / (lam_denom + 1e-15)
        w = C_inv.dot(tmp + lam*ones)
        weights[i, nbors_idx[i]] = w.reshape(-1)

    return weights


def Y_(Weights, d):
    """
    Calculate embedded coordinates in target space

    Parameters
    ----------
    Weights: (n,n) array, weight matrix
    d: dimensionality of target space

    Returns
    -------
    Y: (n,d) array
        Embedded coordinates in target space
    """
    n, p = Weights.shape
    I = np.eye(n)
    m = (I-Weights)
    M = m.T.dot(m)

    eigvals, eigvecs = eigh(M)
    ind = np.argsort(np.abs(eigvals))

    return(eigvecs[:, ind[1:d+1]])


class LocalLinearEmbedding:
    def __init__(self, n_dim=2,
                 n_neighbors=None,
                 epsilon=None):
        assert(n_neighbors != None or epsilon != None)

        self.dist_func = nearest_neighbor_distance if n_neighbors != None else fixed_radius_distance
        self.n_neighbors = n_neighbors
        self.n_dim = n_dim
        self.epsilon = epsilon

    def fit(self, X):
        # Select neighbors
        if self.epsilon is not None:
            _, nbors = self.dist_func(X, self.epsilon)
        elif self.n_neighbors is not None:
            _, nbors = self.dist_func(X, self.n_neighbors)

        # Reconstruct with linear weights
        Weights = get_weights(X, nbors, None)

        # Map to embedded coordinates
        Y = Y_(Weights, self.n_dim)
        return Y.T
