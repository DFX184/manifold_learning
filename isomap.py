import warnings
from mds import *
import numpy as np
from utils import *
import matplotlib.pyplot as plt

class NotExistPath(UserWarning):
    pass


class IsoMap:
    def __init__(self,
                 n_dim=2,
                 n_neighbors=None,
                 epsilon=None):
        self.n_neighbors = n_neighbors
        self.n_dim = n_dim
        self.epsilon = epsilon

        self.choose = -1
        if self.epsilon == None and self.n_neighbors != None:
            self.dist_func = nearest_neighbor_distance
            self.choose = 0
        elif self.epsilon != None and self.n_neighbors == None:
            self.dist_func = fixed_radius_distance
            self.choose = 1
        else:
            self.dist_func = nearest_neighbor_distance
            self.choose = 0

    def fit(self, x,
            cmd_method='cmds',
            lr=1.0):
        """
        ISOMAP

        Parameters
        ----------
        x: (d,n) array, where n is the number of points and n is its dimensionality.
        n_components: dimentionality of target space
        n_neighbors: the number of neighourhood
        epsilon: fixed radius
        dist_func: function for calculating distance matrix

        Returns
        -------
        Y: (d,n) array. Embedded coordinates from cmds in Step 3.
        dist_mat: (n,n)array. Distance matrix made in Step 1.
        predecessors: predecessors from "shortest_path" function in Step 2.
        """

        n_points = x.shape[1]

        params = self.epsilon if self.epsilon != None else self.n_neighbors

        neb_dist, neb_index = self.dist_func(x, params)

        adjmatrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j, idx in enumerate(neb_index[i]):
                adjmatrix[i][idx] = neb_dist[i][j]

        dist_mat, predecessors = shortest_path(csr_matrix(
            adjmatrix), directed=False, return_predecessors=True)

        try:
            model = MDS(n_dim=self.n_dim, input_type='distance')

            Y = model.fit(dist_mat, method=cmd_method, lr=lr)
        except:
            warnings.warn(
                "There is a broken circuit in the diagram", NotExistPath)
            model = MDS(n_dim=self.n_dim, input_type='distance')
            Y = model.fit(dist_mat, method=cmd_method, lr=lr)

        self.distance_matrix = dist_mat

        return Y


if __name__ == '__main__':
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt
    n_points = 1000
    data_s_roll, color = make_swiss_roll(n_points)
    data_s_roll = data_s_roll.T
    data_s_roll.shape
    nebs = [i*8 for i in range(2, 18)]
    plt.figure(figsize=(18, 18))
    for i, n in enumerate(nebs):
        plt.subplot(4, 4, i+1)
        plt.title(f'neighbor= {n}')
        Z = IsoMap(2, n_neighbors=n).fit(data_s_roll)
        Z = Z.T
        plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.savefig('isomap.jpg')
    plt.show()


