import glob
import numpy as np
import os
import pickle
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.stats as stats
import sys
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from matplotlib import pyplot as plt
import corner

# from emulator import *
# from HMF_piecewise import *
# from likelihood import *
from pca import *
from make_fits import *


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


def predict(gp, Omega_m, M_nu, A_s):
    gp_input = np.array([Omega_m, M_nu, A_s]).reshape(1, -1)
    return gp.predict(gp_input)


def test(ind, X, Y, kernel):
    X_train = np.delete(X, ind, axis=0)
    Y_train = np.delete(Y, ind, axis=0)
    gp_test = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    gp_test.fit(X_train, Y_train)
    return gp_test


def integrate_HMF(M_logspace, HMF, num_bins=20):
    assert M_logspace.size == HMF.size
    size = HMF.size // num_bins
    res = np.zeros((num_bins, 1))

    for i in range(num_bins):
        idx_lo = size*i
        idx_hi = size*(i+1)+1
        res[i] = integrate.simps(
            1e9*HMF[idx_lo:idx_hi] /
            M_logspace[idx_lo:idx_hi], M_logspace[idx_lo:idx_hi]
        )

    return res


if __name__ == '__main__':
    # Load and sort data

    emulator = RedshiftTester(M_low=12, M_high=16)
    print("Input shape: ", emulator.X.shape)

    redshift = 0.
    redshift_ind = np.where(emulator.X[:, 3] == redshift)
    # redshift_n = -2
    # redshifts = np.load("redshifts.npy")
    # redshift = redshifts[redshift_n]
    # redshift_ind = np.where(np.isclose(emulator.X[:,3],redshifts[redshift_n],atol=1e-3))
    X = emulator.X[redshift_ind]
    Y = emulator.Y[redshift_ind]
    X, Y = sort_data(X, Y)

    filelist = glob.glob('./hmf_params/z=0/*.npy')
    filelist.sort()

    params = np.ones((len(filelist), 23))
    for i in range(len(filelist)):
        params[i] = np.load(filelist[i])

    ind_h, ind_v = np.where(params[:, 1:] > 0)
    params[ind_h, ind_v+1] = 0

    Mpiv = 1e14
    M_arr_full = np.logspace(13, 15.5, 1001)

    # Create fits from the fits made by make_fits.py
    Y_fit = np.ones((1001, params.shape[0]))
    for i in range(params.shape[0]):
        Y_fit[:, i], _ = get_HMF_piecewise(
            params[i][1:], reg_bins=19, offset=0, Mpiv=1e14)

    HMF_mean = np.average(Y_fit, axis=1).reshape(-1, 1)
    for i in range(Y_fit.shape[1]):
        input_HMFs, = plt.semilogx(np.logspace(
            13, 15, 1001), Y_fit[:, i], color='dimgrey')
    blue_line, = plt.semilogx(np.logspace(
        13, 15, 1001), HMF_mean, color='blue')

    input_HMFs.set_label("Input HMFs $z = 0$")
    blue_line.set_label("Mean HMF")
    plt.legend(handles=[input_HMFs, blue_line], loc='lower left')
    plt.title("HMFs at $z = 0$")
    plt.xlabel("Mass [$M_\odot / h$]")
    plt.ylabel("ln(HMF)")
    plt.show()

    Y_res = Y_fit - np.broadcast_to(HMF_mean, (HMF_mean.size, len(filelist)))

    for i in range(Y_res.shape[1]):
        residual_lines, = plt.plot(np.logspace(
            13, 15, 1001), Y_res[:, i], color='dimgrey')

    residual_lines.set_label("Residual around mean")
    plt.legend(handles=[residual_lines], loc='upper left', prop={'size': 13})
    plt.xlabel("Mass [$M_\odot / h$]")
    plt.ylabel("ln(HMF)")
    plt.show()

    # Calculate PCA and plot the first four eigenvectors
    pca = Pca.calculate(Y_res)
    handles = []
    for i in range(4):
        globals()["evector" + str(i+1)
                  ], = plt.semilogx(np.logspace(13, 15, 1001), pca.u[:, i])
        globals()["evector" + str(i+1)].set_label("EV #" + str(i+1))
        handles.append(globals()["evector" + str(i+1)])

    plt.legend(handles=handles, loc='upper left')
    plt.title("The first four eigenvectors (over $99.9995 \%$ explained variance)")
    plt.xlabel("Mass [$M_\odot / h$]")
    plt.ylabel("ln(HMF)")
    plt.show()

    # Load and plot MCMC samples created by mcmc.py
    samples = np.load("./mcmc_samples/n=27/samples.npy")

    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    labels = ["$w_1$", "$w_2$", "$w_3$", "$w_4$"]
    for i in range(4):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    # axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(100))

    axes[-1].set_xlabel("step number")
    # plt.savefig("mcmc1.png", dpi=320)
    # axes[0].set_title("MCMC on the weights")
    plt.show()

    flat_samples = np.load("./mcmc_samples/n=27/samples_flat.npy")

    fig = corner.corner(
        flat_samples, labels=labels, truths=np.dot(
            np.diagflat(pca.s), pca.v)[:4, 0]
    )

    # plt.savefig("corner.png", dpi=320)
    plt.show()

    # Extract and interpolate weights for the first four eigenvectors
    # using Gaussian Processes.
    PC_weights = np.zeros((len(filelist), 4))
    for i in range(len(filelist)):
        path = "./mcmc_samples/n=%i/samples_flat.npy" % i
        samples = np.load(path)
        PC_weights[i] = np.mean(samples, axis=0)

    X_GP = X[::20, :3]
    Y_GP = PC_weights[:, :].reshape(-1, 4)

    kernel = RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    gp.fit(X_GP, Y_GP)

    # predict(0.09805, 0.3, 1.9)

    n = 23

    # kernel = RBF() + 1.5*Matern(length_scale=.1)
    kernel = RBF()

    X_space = np.logspace(13.0625, 15.4375, 1001)
    gp_test = test(n, X_GP, Y_GP, kernel=RBF(length_scale=5))
    W_predicted = gp_test.predict(X_GP[n].reshape(1, -1)).flatten()
    Y_predicted = np.zeros(1001)
    Y_experimental = np.zeros(1001)
    for i in range(4):
        Y_predicted += W_predicted[i] * pca.u[:, i]
        Y_experimental += Y_GP[n, i] * pca.u[:, i]
    print(W_predicted)
    l_exp, = plt.semilogx(X_space, Y_experimental)
    l_pred, = plt.semilogx(X_space, Y_predicted, alpha=0.7)
    l_pca, = plt.semilogx(X_space, Y_res[:, n])
    plt.title(r"GP for $(\Omega_m, M_{\nu}, A_s) = (%f, %f, %f)$" % tuple(
        X_GP[n].tolist()))
    plt.xlabel("$M$", size=15)
    plt.ylabel(r"$\ln(\frac{dn}{d\ln M})$", size=15)
    plt.legend([l_exp, l_pred, l_pca], ["Experimental", "Predicted", "PCA"])
    plt.show()

    # Show non-normalized predictions
    l_exp, = plt.semilogx(X_space, HMF_mean+Y_experimental.reshape(-1, 1))
    l_pred, = plt.semilogx(X_space, HMF_mean+Y_predicted.reshape(-1, 1))
    l_pca, = plt.semilogx(X_space, HMF_mean+Y_res[:, n].reshape(-1, 1))
    plt.title(r"GP for $(\Omega_m, M_{\nu}, A_s) = (%f, %f, %f)$" % tuple(
        X_GP[n].tolist()))
    plt.xlabel("$M$", size=15)
    plt.ylabel(r"$\ln(\frac{dn}{d\ln M})$", size=15)
    plt.legend([l_exp, l_pred, l_pca], ["Experimental", "Predicted", "PCA"])
    plt.show()

    HMF_exp = np.e**(HMF_mean + Y_experimental.reshape(-1, 1))
    HMF_pred = np.e**(HMF_mean + Y_predicted.reshape(-1, 1))
    # HMF_PCA = np.e**(Y_fit[:, n])

    # Show integrated prediction
    M_logspace = np.logspace(13, 15.5, 1001)
    N_exp = integrate_HMF(M_logspace, HMF_exp.flatten())
    N_pred = integrate_HMF(M_logspace, HMF_pred.flatten())
    N_data = Y[n*20:(n+1)*20]
    # N_PCA = integrate_HMF(M_logspace, HMF_PCA.flatten())
    l_pred, = plt.semilogx(np.logspace(13.0625, 15.4375, 20), N_pred)
    l_exp, = plt.semilogx(np.logspace(13.0625, 15.4375, 20), N_exp)
    # l_pca, = plt.semilogx(np.logspace(13.0625, 15.4375, 20), N_PCA)
    l_data, = plt.semilogx(10**(X[:, 4][n*20:(n+1)*20]), N_data)
    # plt.semilogx(np.logspace(13.0625, 15.4375, 20), integrate_params(get_HMF_piecewise(np.load(filelist[n])[1:], reg_bins=19, offset=0, Mpiv=1e14)[1]))
    plt.title(r"Count for $(M_{\nu}, \Omega_m, A_s) = (%f, %f, %f)$" % tuple(
        X_GP[n].tolist()))
    plt.xlabel("$M$", size=15)
    plt.ylabel("$N$", size=15)
    # plt.legend([l_exp, l_pred, l_pca, l_data], ["MCMC","GP", "PCA", "Data"])
    plt.legend([l_pred, l_exp, l_data], ["GP", "MCMC Weights", "Data"])
    plt.show()
