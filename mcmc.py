import emcee
import glob
import numpy as np
import scipy.integrate as integrate
import scipy.linalg
import time
import os
import pickle


# from emulator import *
# from HMF_fit import *
# from HMF_piecewise import *
# from likelihood import *
# from pca import *

class HaloEmulator:

    def __init__(self, path='./hmf/', mass_type='M200c'):
        self.path = path
        self.mass_type = mass_type
        self.point_list = HaloEmulator.extract_data(path, mass_type)
        self.X, self.Y = HaloEmulator.convert_data(self.point_list)

    @staticmethod
    def extract_data(path, mass_type):
        filelist = glob.glob(os.path.join(path, 'dndm_' + mass_type + '*.pkl'))
        points = []
        for filename in filelist:
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            points.append(HaloEmulator.reshape_data(data))
        return np.concatenate(points, axis=0)

    @staticmethod
    def convert_data(data):
        size = len(data)
        X = np.zeros((size, data[0].size-1))
        Y = np.zeros((size, 1))
        for i in range(size):
            X[i] = np.copy(data[i][:5])
            Y[i][0] = data[i][5]
        return (X, Y)

    @staticmethod
    def reshape_data(point_list):
        theta, a, m, counts = np.array(
            point_list[0]), point_list[1], point_list[2], point_list[3]
        z = 1/a - 1
        theta_reshaped = np.broadcast_to(theta, counts.shape + theta.shape)
        z_reshaped = np.moveaxis(np.broadcast_to(z, m.shape + z.shape), 1, 0,)
        m_reshaped = np.broadcast_to(m, z.shape + m.shape)
        back_half_of_array = np.stack((z_reshaped, m_reshaped, counts), axis=2)
        return np.concatenate((theta_reshaped, back_half_of_array), axis=2).reshape(-1, 6)


class RedshiftTester(HaloEmulator):

    def __init__(self, path='./hmf/', mass_type='M200c', M_low=0, M_high=None, n_chunks=None, redshift=None):
        super().__init__(path, mass_type)
        self.M_low = M_low
        self.M_high = M_high
        self.n_chunks = n_chunks
        self.redshift = redshift
        self.X, self.Y = RedshiftTester.set_limits(
            self.X, self.Y, self.M_low, self.M_high, self.n_chunks)

    @staticmethod
    def set_limits(X, Y, M_low=0, M_high=None, n_chunks=None):
        limits = []
        n = 0

        if M_high is None:
            limits = np.where((X[:, 4] >= M_low))[0]
            M_high = X[:, 4].max()
        else:
            limits = np.where((X[:, 4] >= M_low) & (X[:, 4] <= M_high))[0]

        try:
            n = np.multiply(n_chunks, int((M_high - M_low) / 0.05))
            if isinstance(n, np.int64):
                return(X[limits][:n], Y[limits][:n])
            elif isinstance(n, np.ndarray):
                assert n.shape == (2,), "[n_chunks] has to have size 2!"
                return(X[limits][n[0]:n[-1]], Y[limits][n[0]:n[-1]])
            else:
                raise IOError("Input mismatch!")
        except TypeError:
            if n_chunks is None:
                return(X[limits], Y[limits])
            else:
                raise IOError(
                    "[n] should be either NoneType, an integer, or a size-2 tuple!")


# Credits to Dylan Cromer for the code on PCA:
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


def get_HMF_piecewise(p, **kwargs):

    M_arr_full = np.logspace(13, 15.5, 1001)
    ln_out_HMF = np.zeros(1001)
    res = []

    chunk_size = 50

    reg_bins = kwargs['reg_bins']
    offset = kwargs['offset']

    N_bin = reg_bins+1
    edge_der = np.zeros(N_bin+1)
    Mpiv = kwargs['Mpiv']
    ln_M_Mpiv = np.log(M_arr_full/Mpiv)

    if len(p) != reg_bins+3:
        print("Wrong number of params")
        return -np.inf

    # ln_HMF = a + b*np.log(M_arr_full) + c*np.log(M_arr_full)^2
    # dln_HMF/dlnM = b + 2*c*np.log(M_arr_full)
    # d^2ln_HMF/dlnM^2 = 2*c

    # Pivot bin: 4th bin
    a, b, c = p[0], p[1], p[5]
    idx_lo, idx_hi = 3*chunk_size+offset, 4*chunk_size+offset
    ln_out_HMF[idx_lo:idx_hi] = a + b * \
        ln_M_Mpiv[idx_lo:idx_hi] + c * ln_M_Mpiv[idx_lo:idx_hi]**2
    edge_der[3] = b + 2*c*ln_M_Mpiv[idx_lo]
    edge_der[4] = b + 2*c*ln_M_Mpiv[idx_hi]
    res.append((4, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Go "left"
    for i in range(0, 3)[::-1]:
        idx_lo = i*chunk_size+offset
        idx_hi = (i+1)*chunk_size+offset
        c = p[i+2]
        b = edge_der[i+1] - 2*c*ln_M_Mpiv[idx_hi+1]
        a = ln_out_HMF[idx_hi+1] - b * \
            ln_M_Mpiv[idx_hi+1] - c*ln_M_Mpiv[idx_hi+1]**2
        edge_der[i] = b + 2*c*ln_M_Mpiv[idx_lo]
        ln_out_HMF[idx_lo:idx_hi] = a + b * \
            ln_M_Mpiv[idx_lo:idx_hi] + c*ln_M_Mpiv[idx_lo:idx_hi]**2

        res.append((i+1, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    res.reverse()

    # Extend first bin (no curvature c=0)
    idx_lo = 0
    idx_hi = offset
    b = edge_der[0]
    a = ln_out_HMF[idx_hi+1] - b*ln_M_Mpiv[idx_hi+1]
    ln_out_HMF[idx_lo:idx_hi] = a + b*ln_M_Mpiv[idx_lo:idx_hi]

    if offset != 0:
        res.insert(0, (0, a, b, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Go "right"
    for i in range(4, N_bin-1):
        idx_lo = i*chunk_size+offset
        idx_hi = (i+1)*chunk_size+offset
        c = p[i+2]
        b = edge_der[i] - 2*c*ln_M_Mpiv[idx_lo]
        a = ln_out_HMF[idx_lo-1] - b * \
            ln_M_Mpiv[idx_lo-1] - c*ln_M_Mpiv[idx_lo-1]**2
        edge_der[i+1] = b + 2*c*ln_M_Mpiv[idx_hi]
        ln_out_HMF[idx_lo:idx_hi] = a + b * \
            ln_M_Mpiv[idx_lo:idx_hi] + c*ln_M_Mpiv[idx_lo:idx_hi]**2

        res.append((i+1, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    # Extend last bin with same curvature
    idx_lo = (N_bin-1)*chunk_size+offset
    idx_hi = len(ln_out_HMF)-1
    c = p[-1]
    b = edge_der[-2] - 2*c*ln_M_Mpiv[idx_lo]
    a = ln_out_HMF[idx_lo-1] - b*ln_M_Mpiv[idx_lo-1] - c*ln_M_Mpiv[idx_lo-1]**2
    edge_der[-1] = b + 2*c*ln_M_Mpiv[idx_hi]
    ln_out_HMF[idx_lo:idx_hi+1] = a + b * \
        ln_M_Mpiv[idx_lo:idx_hi+1] + c*ln_M_Mpiv[idx_lo:idx_hi+1]**2

    if idx_lo < idx_hi:
        res.append((N_bin, a, b, c, M_arr_full[idx_lo], M_arr_full[idx_hi]))

    return (ln_out_HMF, res)


def sort_data(X, Y):
    ''' Sorts input by mu_n, in ascending order.'''
    # Concatenate X and Y to make sorting easier.
    M = np.concatenate((X, Y), axis=1)
    M_new = M[M[:, 0].argsort()]

    # Sorting by mu_n messed up the order within each cosmology -
    # mass is no longer sorted in ascending order. This is fixed in the
    # for-loop below.
    for i in range(M_new.shape[0]//20):
        idx_lo = 20*i
        idx_hi = 20*(i+1)
        chunk = M_new[idx_lo:idx_hi]
        M_new[idx_lo:idx_hi] = chunk[chunk[:, 4].argsort()]

    X_new = M_new[:, :5]
    Y_new = np.expand_dims(M_new[:, 5], axis=1)

    return X_new, Y_new


def load_cov_matrices(path):
    pic_in = open(path, "rb")
    cov_matrices = pickle.load(pic_in, encoding="latin1")

    return cov_matrices


def remove_zeros(C):
    idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
    for i in idx[::-1]:
        C = np.delete(C, i, axis=0)
        C = np.delete(C, i, axis=1)
    # print("Covariance matrix shape: ", C.shape)
    return C, idx


a = HaloEmulator()
b = RedshiftTester(M_low=12, M_high=16)
print("Input shape: ", b.X.shape)

redshift = 0.
redshift_ind = np.where(b.X[:, 3] == redshift)
X = b.X[redshift_ind]
Y = b.Y[redshift_ind]
X, Y = sort_data(X, Y)

filelist = glob.glob('./hmf_params/*.npy')
filelist.sort()
filelist

# Params is a numpy array of shape (102, 23).
params = np.ones((len(filelist), 23))
for i in range(params.shape[0]):
    params[i] = np.load(filelist[i])

# Following the constraint c_i <= 0, remove positive entries.
ind_h, ind_v = np.where(params[:, 1:] > 0)
params[ind_h, ind_v+1] = 0

Mpiv = 1e14

# Put HMF data into a single matrix.
Y_fit = np.ones((1001, params.shape[0]))

for i in range(params.shape[0]):
    Y_fit[:, i], _ = get_HMF_piecewise(
        params[i][1:], reg_bins=19, offset=0, Mpiv=1e14)

# Calculate the residuals and conduct PCA analysis.
HMF_mean = np.average(Y_fit, axis=1).reshape(-1, 1)
Y_res = Y_fit - np.broadcast_to(HMF_mean, (HMF_mean.size, len(filelist)))
pca = Pca.calculate(Y_res)


def fit_weights(weights, evectors):
    '''Returns HMF that is based on the given N weights and the first N 
    given eigenvectors. Returns a Numpy array.
    Prerequisites: len(evectors) >= len(weights).'''
    res = np.zeros((1001,))
    for i in range(len(weights)):
        res += weights[i] * evectors[:, i]
    return res


def integrate_dn_dlnM(HMF):
    '''Integrates the given HMF. Returns a (20,1) array of integrated values 
    (not rounded).'''
    # 20 bins.
    M_logspace = np.logspace(13, 15.5, 1001)
    vals = np.zeros((20, 1))
    for i in range(20):
        ind_lo = 50*i
        ind_hi = 50*(i+1)+1
        x = M_logspace[ind_lo:ind_hi]
        dn_dlnM = np.exp(HMF[ind_lo:ind_hi])
        y = 1e9 * dn_dlnM / x
        vals[i] = integrate.trapz(y, x)
    return vals


def log_likelihood_mcmc(weights, N_sim, n):
    '''The cosmology is provided by the number in the (sorted) list of 
    cosmologies, starting at [0., 0.3, 2.1].'''
    filelist_mat = glob.glob("./covmat/covmat_M200c_*.pkl")
    filelist_mat.sort()
    cov_matrix = load_cov_matrices(filelist_mat[n])[-1]
    C, idx = remove_zeros(cov_matrix)

    N_weights = integrate_dn_dlnM(
        HMF_mean[:, 0] + fit_weights(weights, pca.u))

    idx_lo = 20*n
    idx_hi = 20*(n+1)

    return -likelihood_sim(N_weights, N_sim[idx_lo:idx_hi], C, idx)


def likelihood_sim(N_sim, N_model, C, idx):
    assert N_sim.shape == N_model.shape
    assert C.shape[0] == C.shape[1]
    # assert len(idx) == N_sim.size - C[:, 0].size
    D = N_sim - N_model
#    print(" Product: ", D.T @ np.linalg.inv(C) @ D)
    for i in idx[::-1]:
        D = np.delete(D, i, axis=0)
    size = C.shape[0]
    D = D[:size]
    # print(D)
    # print(D.shape, C.shape)
    return (1/2) * D.T.dot(np.linalg.inv(C)).dot(D)


def log_prior(weights):
    for w in weights:
        if -700. < w < 700.:
            return 0.
    return -np.inf


def log_probability(weights, **kwargs):
    N_sim = kwargs['N_sim']
    n = kwargs['n']
    lp = log_prior(weights)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_mcmc(weights, N_sim, n)


def mcmc(N_sim, n, Nsteps=10000, ndim=4, nwalkers=32,
         log_probability=log_probability):
    # Initial position is a Gaussian ball around the PCA weights.
    initial_weights = np.dot(np.diagflat(pca.s), pca.v)[:4, n]
    p0 = np.broadcast_to(initial_weights, (nwalkers, initial_weights.size))
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC; calculate running time.
    tic = time.perf_counter()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, kwargs=({
                                    'N_sim': N_sim, 'n': n}))
    sampler.run_mcmc(pos, Nsteps, progress=True)
    toc = time.perf_counter()

    path = "./mcmc_samples/n=" + str(n) + "/"
    # Create the direction if it doesn't exist.
    if not os.path.exists(path):
        os.makedirs(path)
    samples_flat = []

    try:
        # Obtain tau and calculate burnin and thin values.
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))

        # Obtain the chains and the log probabilities.
        samples_flat = sampler.get_chain()
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(
            discard=burnin, flat=True, thin=thin)
        log_prior_samples = sampler.get_blobs(
            discard=burnin, flat=True, thin=thin)

        # Save everything.
        np.save(path + "samples", sampler.get_chain())
        np.save(path + "samples_flat", samples)
        np.save(path + "log_prob_samples", log_prob_samples)
        np.save(path + "log_prior_samples", log_prior_samples)
    # If unable to obtain tau, simply record the chains without discarding any steps
    # nor thinning the chain.
    except emcee.autocorr.AutocorrError:
        samples = sampler.get_chain()
        log_prob_samples = sampler.get_log_prob(flat=True)
        log_prior_samples = sampler.get_blobs(flat=True)

        np.save(path + "samples_no_tau", samples)
        np.save(path + "log_prob_samples_no_tau", log_prob_samples)
        np.save(path + "log_prior_samples_no_tau", log_prior_samples)

    print("Time: %f seconds." % (toc-tic))

    return samples, samples_flat, log_prob_samples, log_prior_samples


if __name__ == '__main__':
    ndim = 4
    nwalkers = 32
    Nsteps = 15000

    mcmc_samples = []

    print(len(filelist))
    for i in [89, 92]:
        print(filelist[i])
        print("Current sample: %d" % i)
        _, samples, _, _ = mcmc(Y, i, Nsteps=Nsteps,
                                ndim=ndim, nwalkers=nwalkers, log_probability=log_probability)
        mcmc_samples.append(samples)

    # samples_new = sampler.get_chain(discard=burnin, flat=True, thin=thin)
