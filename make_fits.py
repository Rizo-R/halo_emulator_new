import random
import glob
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
import sys
import time


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


def remove_zeros(C):
    '''Removes all rows and columns with 0 in their diagonal entries (e.g. 
    if the i-th diagonal entry is 0 then both i-th row and i-th column are 
    removed. This is done to keep the covariance matrix positive semidefinite.'''
    idx = np.argwhere(np.all(C == 0, axis=0) | np.all(C == 0, axis=1))
    for i in idx[::-1]:
        C = np.delete(C, i, axis=0)
        C = np.delete(C, i, axis=1)
    # Remove repeating CONSECTUIVE rows/columns
    for i in reversed(range(C.shape[0]-1)):
        if np.all(C[i] == C[i+1]) and np.all(C[:, i] == C[:, i+1]):
            C = np.delete(C, i, axis=0)
            C = np.delete(C, i, axis=1)
    print("Covariance matrix shape: ", C.shape)
    return C, idx


def is_pos_def(x):
    # Some eigenvalues ended up being very close to zero so had to modify this
    return np.all(np.linalg.eigvals(x) > -1e-10)
    # return np.all(np.linalg.eigvals(x) > 0)


def get_HMF_piecewise(p, **kwargs):
    '''Recover all a,b,c parameters for bins using continuity equations used
    in Section 3.1 of the paper by Bocquet et al. (https://arxiv.org/pdf/2003.12116.pdf)'''

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


def f_polynomial(M, a, b, c):
    return a + b * np.log(M/Mpiv) + c * np.log(M/Mpiv)**2


def f_integrand(M, a, b, c):
    return 1e9 * (1/M) * np.e**(f_polynomial(M, a, b, c))


def integrate_dn_dlnM(param_list, zeroth_bin=False):
    '''Integrate bins given the parameter list, as well as bounds (in mass)
    of each bin.'''
    res = np.ones((len(param_list), 1))

    for i in range(res.size):
        # 0th bin integration
        if zeroth_bin and i == 0:
            a, b, M_min, M_max = param_list[i][1:]
            res[i] = integrate.quad(
                f_integrand, M_min, M_max, args=(a, b, 0))[0]
        # i-th bin integration
        else:
            a, b, c, M_min, M_max = param_list[i][1:]
            res[i] = integrate.quad(
                f_integrand, M_min, M_max, args=(a, b, c))[0]

    return res

# pr = False


def likelihood_sim(N_sim, N_model, C, idx):
    '''See Eq.16 in Bocquet et al.'''
    assert N_sim.shape == N_model.shape
    assert C.shape[0] == C.shape[1]
    # assert len(idx) == N_sim.size - C[:, 0].size
    D = N_sim - N_model
#    print(" Product: ", D.T @ np.linalg.inv(C) @ D)
    for i in idx[::-1]:
        D = np.delete(D, i, axis=0)
    size = C.shape[0]
    D = D[:size]
    # if pr:
    #     # print(N_sim, N_model)
    #     # print(D)
    #     # print(C)
    #     # print(D.T.dot(D))
    #     print(np.where(C != C.T))
    #     print(np.linalg.eigvals(np.linalg.inv(C)))
    #     print(is_pos_def(np.linalg.inv(C)))
    #     print(D.T.dot(np.linalg.inv(C)).dot(D))
    # print(D)
    # print(D.shape, C.shape)
    return (1/2) * D.T.dot(np.linalg.inv(C)).dot(D)


def likelihood_var(lambd, c_arr):
    '''See Eq.17 in Bocquet et al.'''
    c_sum = 0
    for i in range(c_arr.size-1):
        c_sum += (c_arr[i] - c_arr[i+1])**2
    c_sum /= (2 * lambd**2)

    return (c_arr.size - 1) * np.log(lambd) + c_sum


def likelihood_total(params, overwrite=True):
    '''See Eq.18 in Bocquet et al. 
    params must be passed in as [lambd, a4, b4, c1, c2, c3,..., cN],
    where a,b,c are the parameters for the corresponding bin, and lambd 
    is a hyperparameter mentioned in the paper.'''
    global toc
    global likelihood_min
    global pr

    if params.size == 0 or params.shape == (1, 23):
        fname = "./hmf_params_final/current_optimizer_" + \
            m_nu + '_' + o_m + '_' + A_s + '_' + z + ".npy"
        params = np.load(fname)

    lambd = params[0]
    p = params[1:]
    # print(p)
    _, hmf_piecewise_params = get_HMF_piecewise(
        p, reg_bins=19, Mpiv=Mpiv, offset=0)
    N_model = integrate_dn_dlnM(hmf_piecewise_params, zeroth_bin=False)

    # .item() is needed to extract the value from a size-1 array
    res = (likelihood_sim(Y_curr, N_model, C, idx) +
           likelihood_var(lambd, params[3:])).item()

    cost_overrun = 0

    for c in params[2:]:
        if c > 0.:
            cost_overrun += (1000*c)**8

    if params[0] < 1:
        cost_overrun += (1000*(1-params[0]))**8

    if not overwrite:
        print(N_model)
        print("overrun: ", cost_overrun)

    if overwrite and res + cost_overrun < likelihood_min:
        pr = True
        print("var: ", likelihood_var(lambd, params[3:]))
        print("sim: ", likelihood_sim(Y_curr, N_model, C, idx))
        pr = False
        likelihood_min = res
        fname = "./hmf_params_final/current_optimizer_" + \
            m_nu + '_' + o_m + '_' + A_s + '_' + z + ".npy"
        np.save(fname, params)
        print("Saved to ", fname)

    toc = time.perf_counter()
    return res + cost_overrun


def remove_positives(params):
    # Remove positive entries for the c-parameter.
    # ind_h, ind_v = np.where(params[:, 1:] > 0)
    # params[ind_h, ind_v+1] = 0
    ind = np.where(params[1:] > 0)[0]
    params[ind+1] = 0
    return params


def set_initial_guess():
    arr = np.tile(np.linspace(-1,  -5, 4), 5)
    params = np.concatenate((np.ones((3, 1)), arr[:, np.newaxis]), axis=0)
    params[0] = 1.01
    params[1] = -13
    params[2] = -1.2

    return params


def load_cov_matrices(path):
    pic_in = open(path, "rb")
    cov_matrices = pickle.load(pic_in, encoding="latin1")

    return cov_matrices


if __name__ == '__main__':
    emulator = RedshiftTester(M_low=13, M_high=16, path="./hmf/")
    print("Input shape: ", emulator.X.shape)

    Mpiv = 1e14

    # Sort X and Y
    mask = np.lexsort((emulator.X[:, 4], emulator.X[:, 3], emulator.X[:, 0]))
    X = emulator.X[mask]
    Y = emulator.Y[mask]

    # Need 0 < z <= 0.5 - roughly the first 10 redshifts
    mask_lower = np.where(X[:, 3] > 0)
    X = X[mask_lower]
    Y = Y[mask_lower]
    mask_higher = np.where(X[:, 3] <= 0.5)
    X = X[mask_higher]
    Y = Y[mask_higher]

    # Have 102 cosmologies, 10 redshifts average each, 20 bins each
    # So X should be (20400,5)
    print(X.shape)

    # Bounds for optimization values
    bnds = [(1., 100.), (-100., 100.), (-100., 100.)]
    for i in range(20):
        bnds.append((-20., 20.))

    # This is where actual fitting happens. The values of n determine which
    # parts of X and Y will be fit (each chunk has 200 values). The current value
    # is 35 because the fit produced for n=35 ended up not being accurate.
    # for n in range(35, 36):
    # for n in range(0, Y.size//200):
    for n in range(10):
        likelihood_min = 1e8
        params = set_initial_guess()

        X_curr_cosmology = X[200*n:200*(n+1)]
        Y_curr_cosmology = Y[200*n:200*(n+1)]
        # Round to 5 decimal places; add decimal places if needed.
        m_nu = "{:.5f}".format(X_curr_cosmology[0, 0])
        o_m = "{:.5f}".format(X_curr_cosmology[0, 1])
        A_s = "{:.4f}".format(X_curr_cosmology[0, 2])

        path = './covmat/covmat_M200c_mnu' + m_nu + \
            '_om' + o_m + '_As' + A_s + '20bins.pkl'
        cov_matrices = load_cov_matrices(path)
        # print(X_curr_cosmology.shape)
        for m in range(10):
            # Each cosmology has EXACTLY 10 redshifts <= 0.5
            if n == 35 and m < 2:
                continue
            X_curr = X_curr_cosmology[20*m:20*(m+1)]
            Y_curr = Y_curr_cosmology[20*m:20*(m+1)]
            C = cov_matrices[-m-2]
            C, idx = remove_zeros(C)
            z = "{:.5f}".format(X_curr[0, 3])

            # Remove negative eigenvalues within floating point error
            # (i.e. -1e-8 < lambda < 0)
            # print(is_pos_def(np.linalg.inv(C)))

            # evals, evecs = np.linalg.eig(C)
            # evals[np.logical_and(evals < 0, evals > -1e-8)] = 0
            # C = evecs @ np.diag(evals) @ evecs.T
            # print((np.linalg.eigvals(np.linalg.inv(C))))

            # print(np.linalg.eigvals(C))
            # print("Symmetric:", np.all(C == C.T))
            # print(is_pos_def(np.linalg.inv(C)))
            # print(np.linalg.inv(C).dot(C))

            # NOTE: this part currently makes the code raise a ValueError.
            # This is where I stopped after Prof. Battaglia told me to check
            # for identical rows and columns. Note that remove_zeros() only
            # removes consecutive identical rows/columns, while this part
            # checks for all pairs.
            # for i in range(C.shape[0]//2):
            #     for j in range(i, C.shape[0]//2):
            #         if np.all(C[i] == C[j]) or np.all(C[:, i] == C[:, j]):
            #             raise ValueError("Identical rows/columns")

            # if not np.all(C == C.T):
            #     float_formatter = "{:.2f}".format
            #     np.set_printoptions(formatter={'float_kind': float_formatter})
            #     raise ValueError()

            # C = cov_matrices[-2]
            # C, idx = remove_zeros(C)

            # print(likelihood_total(params, False))

            if C.size == 0:
                continue

            print("Covariance matrix is positive definite: ", is_pos_def(C))

            if not is_pos_def(C):
                raise Exception("Covariance matrix not positive definite.")

            fname = "./hmf_params_final/current_optimizer_" + \
                m_nu + "_" + o_m + "_" + A_s + '_' + z + ".npy"
            np.save(fname, params)

            tic = time.perf_counter()
            for i in range(10):
                print("Attempt #%i." % (i+1))
                params = np.load(fname)
                try:
                    optimizer = optimize.basinhopping(likelihood_total, params)
                except ValueError:
                    pass

            optimized = np.load(fname)
            optimized = remove_positives(optimized)

            print(optimized.shape)

            print("Finished iteration %i, subiteration %i" % (n, m))
            print("Total likelihood: ", likelihood_total(
                optimized, overwrite=False))
            print("Time: ", str(toc - tic))
            print("\n")
