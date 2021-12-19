import numpy as np
import scipy.linalg

# Credits to Dylan Cromer for the code below:
# https://github.com/dylancromer/pality


class PcData:

    def __init__(self, basis_vectors, weights, explained_variance, u, s, v):
        self.basis_vectors = basis_vectors
        self.weights = weights
        self.explained_variance = explained_variance
        self.u = u
        self.s = s
        self.v = v


class Pca:
    @staticmethod
    def svd_flip(u, v, u_based_decision=True):
        '''
        Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        '''
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v

    @classmethod
    def svd(cls, matrix):
        u, s, v = scipy.linalg.svd(matrix, full_matrices=False)
        u, v = cls.svd_flip(u, v)
        return u, s, v

    @staticmethod
    def basis_vecs_from_svd(u, s):
        n_samps = u.shape[1]
        return (u.dot(np.diagflat(s))) / np.sqrt(n_samps)

    @staticmethod
    def weights_from_svd(v):
        n_samps = v.shape[0]
        return v * np.sqrt(n_samps)

    @staticmethod
    def explained_var_from_s(s, n_samples):
        explained_variance = s**2 / (n_samples-1)
        total_var = explained_variance.sum()
        return explained_variance / total_var

    @classmethod
    def calculate(cls, data):
        u, s, v = cls.svd(data)

        basis_vectors = cls.basis_vecs_from_svd(u, s)

        weights = cls.weights_from_svd(v)

        num_samples = data.shape[0]
        explained_variance = cls.explained_var_from_s(s, num_samples)

        return PcData(basis_vectors, weights, explained_variance, u, s, v)
