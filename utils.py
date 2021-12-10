from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np

def fixed_radius_distance(X, epsilon):
    """
    Calculate epsilon-NN

    Parameters
    ----------
    X: (d,n) array, where n is the number of points and d is its dimension
    epsilon: criterion of selecting neighbors
        Select points as its neighbours if distance < epsilon

    Returns
    -------
    nbrs_dist: (n,k*) array
        It is filled with distances with neighbors. 
        In each row, k* varies according to the number of neighbours
        Each row corresponds to a specific point (row-major order)
    nbrs_idx: (n,k*) array
        It is filled with the indices of neighbors. 
        In each row, k* varies according to the number of neighbours
        Each row corresponds to a specific point (row-major order)
    """
    nbrs_dist = []
    nbrs_idx = []
    data = X.T
    D = euclidean_distances(data, data)
    for i in range(data.shape[0]):
        idx = []
        dst = []
        for j in range(data.shape[0]):
            if D[i][j] < epsilon:
                dst.append(D[i][j])
                idx.append(j)
        nbrs_dist.append(dst)
        nbrs_idx.append(idx)
    return list(map(np.array, nbrs_dist)), list(map(np.array, nbrs_idx))


def nearest_neighbor_distance(X, n_neighbors):
    """
    Calculate K-NN

    Parameters
    ----------
    X: (d,n) array, where n is the number of points and d is its dimension
    n_neighbors: number of neighbors
        Select n_neighbors(k) nearest neighbors

    Returns
    -------
    dist: (n,k) array
        It is filled with distances with neighbors. 
        In each row, k varies according to the number of neighbours
        Each row corresponds to a specific point (row-major order)
    nbrs: (n,k) array
        It is filled with the indices of neighbors. 
        In each row, k varies according to the number of neighbours
        Each row corresponds to a specific point (row-major order)
    """
    distances = euclidean_distances(X.T, X.T)
    neb = NearestNeighbors(metric='euclidean', n_neighbors=n_neighbors)
    neb.fit(X.T)
    nbrs_dist, nbrs_idx = neb.kneighbors(X.T)
    return np.array(nbrs_dist), np.array(nbrs_idx).astype('int')
