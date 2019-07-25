import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn

from autograd import value_and_grad
from scipy.optimize import minimize

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from autograd_transforms import identity_transform, spherical_transform, normalize_transform

from joint_models import *
from marginal_models import *

def simulate_genotype(N, D, maf, p):
    genotype = []
    genotype.append(np.random.binomial(1, maf, D))

    for _ in range(N-1):
        new_snp = np.zeros(D)
        perturb = np.random.binomial(1, p, D)
        new_snp[perturb==1] = genotype[-1][perturb==1]
        new_snp[perturb==0] = 1 - genotype[-1][perturb==0]
        if np.random.rand() > 0.5:
            new_snp = np.mod(new_snp + 1, 2)
        genotype.append(new_snp)

    genotype = np.array(genotype)
    return genotype

def generate_data(N, D, maf, p, T, causal_idx=None):
    G = simulate_genotype(N, D, maf, p)
    Sigma = np.corrcoef(G)

    # pick a causal SNP
    c = np.zeros(N)
    if causal_idx is None:
        causal_idx = int(N/2)
    c[causal_idx] = 1

    # mean statistic
    f = Sigma.dot(c)

    # sample from N(f, Sigma) to get observed statistics
    L = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0])*1e-6)
    y = f + L.dot(np.random.normal(size=f.shape))
    Y = np.stack([f + L.dot(np.random.normal(size=f.shape)) for _ in range(T)]).T
    
    u, s, vt = np.linalg.svd(Sigma)
    X = u * np.sqrt(s)
    return G, X, Y, Sigma


# test ll at SNPs
def test_all_points(function_generator, X, Y, Sigma, M, transform=identity_transform, kwargs={}):
    T = Y.shape[1]
    bound, _unpack_params = function_generator(X, Y, Sigma, M=M, transform=transform, **kwargs)[:2]
    lls = []
    for i in range(X.shape[0]):
        Z = transform.backward(np.tile(X[i][None], (M, 1)))
        v = np.ones((M, T))
        param = np.hstack([Z, v]).flatten()
        lls.append(bound(param))
    return lls

# function to plot results
def plot_ll_at_points(X, Y, Sigma, functions, transform=identity_transform):

    fig, ax = plt.subplots(1, len(functions), sharey=False, figsize=(4 * len(functions), 3))

    N = X.shape[0]
    for i, function in enumerate(functions):
        funcs = globals()[function]
        lls = test_all_points(funcs, X, Y, Sigma, 1, transform)
        ax[i].scatter(np.arange(N), lls)
        ax[i].scatter([int(N/2)], lls[int(N/2)])
        ax[i].set_title(function.split('_')[0])
    return fig


# train
def train(function_generator, X, Y, Sigma, M=1, transform=identity_transform,
          maxiters=1000, params=None, verbose=False, init='kmeans', kwargs={}):
    N, T = Y.shape
    bound, _unpack_params = function_generator(X, Y, Sigma, M=M, transform=transform, **kwargs)[:2]
    lls = []

    if params is None:
        if init is 'kmeans':
            Z_init = transform.forward(KMeans(n_clusters=M).fit(X).cluster_centers_)

        else:
            Z_init = X[np.random.choice(N, M)]
        v_init = np.zeros((M, T))
        param_init = np.hstack([transform.backward(Z_init), v_init]).flatten()

    else:
        param_init = params

    objective = lambda params: -bound(params)
    lls = []
    def callback(params):
        lls.append(-objective(params))
        print("Iter {}: log likelihood {}".format(len(lls), lls[-1]))

    if not verbose:
        callback = None

    params = minimize(value_and_grad(objective), param_init, jac=True, method='CG', callback=callback, options={'maxiter': maxiters})

    _, Z = _unpack_params(params.x)
    _, Z_init = _unpack_params(param_init)
    
    return Z, Z_init, params

def train_and_plot(functions, X, Y, Sigma, M=1, transform=identity_transform, kwargs=None):
    N = X.shape[0]
    if kwargs is None:
        kwargs = [{} for _ in range(len(functions))]

    if not isinstance(transform, list):
        transform = [transform for _ in range(len(functions))]
    
    fig, ax = plt.subplots(1 + M, len(functions), sharey=False, figsize=(4 * len(functions), 3 + 3 * M))
    for i, function in enumerate(functions):
        funcs = globals()[function]
        # IC
        lls = test_all_points(funcs, X, Y, Sigma, M)
        Z, Z_init, params = train(funcs, X, Y, Sigma, M, transform[i])
        distances = cdist(X, Z)
        init_distances = cdist(X, Z_init)
    
        ax[0, i].scatter(np.arange(N), lls)
        ax[0, i].scatter([int(N/2)], lls[int(N/2)])

        for j in range(M):
            ax[j+1, i].scatter(np.arange(N), distances.T[j])            
            ax[j+1, i].scatter([int(N/2)], distances.T[j, int(N/2)])
            ax[j+1, 0].set_ylabel('distance')


        ax[0, i].set_title('_'.join(function.split('_')[:-1]))

    ax[0, 0].set_ylabel('likelihoods')

def train_and_plot_abscorrs(functions, X, Y, Sigma, M=1, transform=identity_transform, kwargs=None):
    N = X.shape[0]
    if kwargs is None:
        kwargs = [{} for _ in range(len(functions))]

    if not isinstance(transform, list):
        transform = [transform for _ in range(len(functions))]
    
    fig, ax = plt.subplots(1 + M, len(functions), sharey=False, figsize=(4 * len(functions), 3 + 3 * M))
    for i, function in enumerate(functions):
        funcs = globals()[function]
        # IC
        lls = test_all_points(funcs, X, Y, Sigma, M)
        Z, Z_init, params = train(funcs, X, Y, Sigma, M, transform[i])
        distances = np.abs(X @ Z.T)
        init_distances = np.abs(X @ Z_init.T)
    
        ax[0, i].scatter(np.arange(N), lls)
        ax[0, i].scatter([int(N/2)], lls[int(N/2)])

        for j in range(M):
            ax[j+1, i].scatter(np.arange(N), distances.T[j])            
            ax[j+1, i].scatter([int(N/2)], distances.T[j, int(N/2)])
            ax[j+1, 0].set_ylabel('|correlation|')


        ax[0, i].set_title('_'.join(function.split('_')[:-1]))

    ax[0, 0].set_ylabel('likelihoods')


def train_and_plot_geodesic(functions, X, Y, Sigma, M=1, transform=identity_transform, kwargs=None):
    N = X.shape[0]
    if kwargs is None:
        kwargs = [{} for _ in range(len(functions))]

    if not isinstance(transform, list):
        transform = [transform for _ in range(len(functions))]
    
    fig, ax = plt.subplots(1 + M, len(functions), sharey=False, figsize=(4 * len(functions), 3 + 3 * M))
    for i, function in enumerate(functions):
        funcs = globals()[function]
        # IC
        lls = test_all_points(funcs, X, Y, Sigma, M)
        Z, Z_init, params = train(funcs, X, Y, Sigma, M, transform[i])
        distances = np.abs(X @ Z.T)
        init_distances = np.abs(X @ Z_init.T)

        ax[0, i].scatter(np.arange(N), lls)
        ax[0, i].scatter([int(N/2)], lls[int(N/2)])

        for j in range(M):
            ax[j+1, i].scatter(np.arange(N), distances.T[j])
            ax[j+1, i].scatter([int(N/2)], distances.T[j, int(N/2)])
            ax[j+1, 0].set_ylabel('geodesic distance')

        ax[0, i].set_title('_'.join(function.split('_')[:-1]))
    ax[0, 0].set_ylabel('likelihoods')

def train_and_plot_vs_init(functions, X, Y, Sigma, M=1, transform=identity_transform, kwargs=None):
    N = X.shape[0]

    # pass empty kwargs if we have none
    if kwargs is None:
        kwargs = [{} for _ in range(len(functions))]
    # if we are using only one transform, dont need to pass a list
    if not isinstance(transform, list):
        transform = [transform for _ in range(len(functions))]
    
    fig, ax = plt.subplots(1 + M, len(functions), sharey=False, figsize=(4 * len(functions), 3 + 3 * M))
    for i, function in enumerate(functions):
        funcs = globals()[function]
        # IC
        lls = test_all_points(funcs, X, Y, Sigma, M)
        Z, Z_init, params = train(funcs, X, Y, Sigma, M, transform[i])
        distances = cdist(X, Z)
        init_distances = cdist(X, Z_init)

        ax[0, i].scatter(np.arange(N), lls)
        ax[0, i].scatter([int(N/2)], lls[int(N/2)])

        for j in range(M):
            ax[j+1, i].scatter(np.arange(N), distances.T[j])
            ax[j+1, i].scatter(np.arange(N), init_distances.T[j])
            
            ax[j+1, i].scatter([int(N/2)], distances.T[j, int(N/2)])
            ax[j+1, i].scatter([int(N/2)], init_distances.T[j, int(N/2)])

        ax[0, i].set_title('_'.join(function.split('_')[:-1]))

    ax[0, 0].set_ylabel('likelihoods')
    ax[1, 0].set_ylabel('distances')
