import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def gradient_descent(D, x0, loss_f, grad_f, lr, tol, max_iter):
    losses = np.zeros(max_iter)
    y_old = x0
    y = x0
    for i in range(max_iter):

        g = grad_f(D, y)

        y = y_old - lr * g
        stress = loss_f(D, y)

        losses[i] = stress
        if stress < tol:
            msg = "\riter: {0}, stress: {1:}".format(i, stress)
            print(msg, flush=True, end="\t")
            losses = losses[:i]
            break

        if i % 50 == 0:
            msg = "\riter: {0}, stress: {1:}".format(i, stress)
            print(msg, flush=True, end="\t")

        y_old = y

    if i == max_iter-1:
        msg = "\riter: {0}, stress: {1:}".format(i, stress)
        print(msg, flush=True, end="\t")

    print('\n')

    return y, losses

'''
input data and output data are col vector (ie : 
    For X \in R^{NxV} X.shape = (V,N) not (N,V)
)
'''

class MDS:
    def __init__(self,
                 n_dim=2,
                 input_type='raw'):

        
        if input_type not in ['distance', 'raw']:
            raise RuntimeError('Not implement type !')

        self.input_type = input_type
        self.n_dim = n_dim

    def fit(self, X,
            method='cmds',  # or stress
            lr=0.5):
        if method == 'cmds':
            return self._cmds(X)
        else:
            return self._stress_based_mds(X, lr=lr)

    def _cmds(self, X):
        """
        Classical(linear) multidimensional scaling (MDS)

        Parameters
        ----------
        X: (d, n) array or (n,n) array
            input data. The data are placed in column-major order. 
            That is, samples are placed in the matrix (X) as column vectors
            d: dimension of points
            n: number of points

        n_dim: dimension of target space

        input_type: it indicates whether data are raw or distance
            - raw: raw data. (n,d) array. 
            - distance: precomputed distances between the data. (n,n) array.
        Returns
        -------
        Y: (n_dim, n) array. projected embeddings.
        evals: (n_dim) eigen values
        evecs: corresponding eigen vectors in column vectors
        """
        Y = None
        evals = None
        evecs = None
        if self.input_type == 'distance':
            D = X
        elif self.input_type == 'raw':
            Xt = X.T
            D = euclidean_distances(Xt, Xt)

        n = len(D)

        H = np.eye(n) - (1/n)*np.ones((n, n))

        D = (D**2).astype(np.float64)

        D = np.nan_to_num(D)
        G = -(1/2) * (H.dot(D).dot(H))

        evals, evecs = np.linalg.eigh(G)

        index = evals.argsort()[::-1]
        evals = evals[index]
        evecs = evecs[:, index]
        evals = evals[:self.n_dim]
        evecs = evecs[:, :self.n_dim]

        self.eigen_vectors = evecs
        self.eigen_values = evals

        Y = np.diag(evals**(1/2)) @ evecs.T
        assert Y.shape[0] == self.n_dim
        return Y

    def _loss_sammon(self, D, y):
        """
        Loss function (stress) - Sammon

        Parameters
        ----------
        D: (n,n) array. distance matrix in original space
            This is a symetric matrix
        y: (d,n) array
            d is the dimensionality of target space.
            n is the number of points.

        Returns
        -------
        stress: scalar. stress
        """
        yt = y.T
        n = D.shape[0]
        Delta = euclidean_distances(yt, yt)
        stress = 0
        for i in range(n):
            f = 0
            s = 0
            for j in range(n):
                s += (D[i, j] - Delta[i, j])**2
                f += Delta[i, j]
            stress += (s/f)
        return stress

    def _grad_sammon(self, D, y):
        """
        Gradient function (first derivative) - Sammonn_dim

        Parameters
        ----------
        D: (n,n) array. distance matrix in original space
            This is a symetric matrix
        y: (d,n) array
            d is the dimensionality of target space.
            n is the number of points.

        Returns
        -------
        g: (k,n) array.
            Gradient matrix. 
            k is the dimensionality of target space.
            n is the number of points.
        """
        D2 = euclidean_distances(y.T, y.T)
        n = len(D)

        def grid(k):
            s = np.zeros(y[:, k].shape)
            for j in range(n):
                if j != k:
                    s += (D2[k, j] - D[k, j])*(y[:, k] - y[:, j])/(D2[k, j])
            return s

        N = 1/np.tril(D, -1).sum()
        g = np.zeros((y.shape[0], n))
        for i in range(n):
            g[:, i] = grid(i)

        return N*g

    def _stress_based_mds(self, x,
                          lr, tol=1e-9, max_iter=6000):
        """
        Stress-based MDS

        Parameters
        ----------
        x: (d,n) array or (n,n) array
            If it is raw data -> (d,n) array
            otherwise, (n,n) array (distance matrix)
            n is the number of points
            d is the dimensionality of original space
        n_dim: dimensionality of target space
        loss_f: loss function
        grad_f: gradient function
        input_type: 'raw' or 'distance'
        init: initialisation method
            random: Initial y is set randomly
            fixed: Initial y is set by pre-defined values
        max_iter: maximum iteration of optimization

        Returns
        -------
        y: (n_dim,n) array. Embedded coordinates in target space
        losses: (max_iter,) History of stress
        """

        # obtain distance
        if self.input_type == 'raw':
            x_t = x.T
            D = euclidean_distances(x_t, x_t)
        elif self.input_type == 'distance':
            D = x
        else:
            raise ValueError('inappropriate input_type')

        # Remaining initialisation
        N = x.shape[1]

        np.random.seed(10)
        # Initialise y randomly
        y = np.random.normal(0.0, 1.0, [self.n_dim, N])

        # calculate optimal solution (embedded coordinates)
        y, _ = gradient_descent(D, y, self._loss_sammon,
                                self._grad_sammon, lr, tol, max_iter)

        return y


