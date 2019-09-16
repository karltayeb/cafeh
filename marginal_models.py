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

"""
Each of these functions takes data as input

returns 
    1. an objective to be MAXIMIZED (they are all likelihoods)
    2. a function for unpacking and transforming flattened
       unconstrained parameters.
    3. any other functions returend after this



CURRENTLY ONLY USING LINEAR KERNEL-- should extend these function
to take kernel as an argument
"""

JITTER = 1e-5

def linear_kernel(X, Y):
    return X @ Y.T

class identity_transform:
    @staticmethod
    def forward(Z_unconstrained):
        return Z_unconstrained

    @staticmethod
    def backward(Z_transformed):
        return Z_transformed

def vfe(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform):
    """
    minimize KL to GP posterior
    this is NOT a model for identifying causal variants
    but I put it here because maybe it will learn informative inducing points?
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        _, Z = _unpack_params(params)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        cov = Sigma + Qff + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        
        A = np.linalg.solve(L, Y)  # N x T
        quad = np.trace(A.T @ A)
        
        trace = np.trace(np.linalg.solve(Sigma + np.eye(N) * jitter, Kff - Qff))
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    
    
    def _optimal_u(params):
        _, Z = _unpack_params(params)
        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        
        Kuu_inv = np.linalg.inv(Kuu + np.eye(M) * jitter)
        Sigma_inv = np.linalg.inv(Sigma + np.eye(N) * jitter)
        S_inv = np.linalg.inv(Kuu + Kfu.T @ Sigma_inv @ Kfu)
        
        mu = Kuu @ S_inv @ Kfu.T @ Sigma_inv @ Y
        A = Kuu @ S_inv @ Kuu
        return mu, A

    def predict(Xnew, params):
        mu, A = _optimal_u(params)
        _, Z = _unpack_params(params)

        # use a linear kernel for now
        Kff = (Xnew @ Xnew.T)
        Kfu = (Xnew @ Z.T)
        Kuu = kernel(Z, Z)
        
        mean = Kfu @ np.linalg.solve(Kuu, mu)
        
        Kuu_inv = np.linalg.inv(Kuu + np.eye(M) * jitter)
        var = Kff - Kfu @ Kuu_inv @ Kfu.T + Kfu @ Kuu_inv @ A @ Kuu_inv @ Kfu.T
        return mean, var

    return bound, _unpack_params, predict, _optimal_u

def fitc_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform):
    """
    functions values are independent given inducing points
    f_i || f_j | u

    with variance given by (K_ii - Q_ii)
    so that the inducing points will tightly regulate nearby points
    but will be less informative far away

    however observations are still correlated through Sigma
    this would probide the advantage of low rank updates

    also without this type of structre, marginalizing out u
    gives the GP prior-- there would be no dependence on our
    inducing points
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = 1e-6
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        _, Z = _unpack_params(params)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        cov = Sigma + Qff + np.diag(np.diag(Kff-Qff)) + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)

        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0

        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def independent_causal_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_A=False):
    """
    we have a vector of independent effect sizes for the causal set ~N(0, sigma^2 I)
    then u = Kuu c

    other than that this is FITC
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = 1e-6
    
    def _unpack_params(params):
        return np.exp(params.reshape(M, D+T)[:, D:]), transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        A, Z = _unpack_params(params)

        if not train_A:
            A = np.ones_like(A)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Kuu = Kuu @ np.diag(A[:, 0]) @ Kuu
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        # take abs because this should always be positive but for numerical reasons
        # it occasionally is not -- improves stability
        cov = Sigma + Qff + np.diag(np.abs(np.diag(Kff-Qff))) + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def independent_inducing_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_A=False, default_variance=1.0):
    """
    here we just say that the u are iid
    its still FITC
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = 1e-6
    def _unpack_params(params):
        return np.exp(params.reshape(M, D+T)[:, D:]), transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        A, Z = _unpack_params(params)

        if not train_A:
            A = np.ones_like(A) * default_variance

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = np.diag(A[:, 0])
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        # take abs because this should always be positive but for numerical reasons
        # it occasionally is not -- improves stability
        cov = Sigma + Qff + np.diag(np.abs(np.diag(Kff-Qff))) + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def dtc_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform):
    """
    function values at X are deterministic given u
    via the conditial mean

    here we ask that Sigma do all the work in explaining variance
    from this mean
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        _, Z = _unpack_params(params)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        cov = Sigma + Qff + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def dtc_independent_causal_functions_old(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_A=False, default_variance=1.0):
    """
    function values at X are deterministic given u
    via the conditial mean

    here we ask that Sigma do all the work in explaining variance
    from this mean
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return np.exp(params.reshape(M, D+T)[:, D:]), transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        A, Z = _unpack_params(params)

        if not train_A:
            A = np.ones_like(A) * default_variance

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)

        Kuu = kernel(Z, Z)
        Kuu = Kuu @ np.diag(A[:, 0]) @ Kuu
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        cov = Sigma + Qff + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def dtc_independent_inducing_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_A=False, default_variance=1.0):
    """
    function values at X are deterministic given u
    via the conditial mean

    here we ask that Sigma do all the work in explaining variance
    from this mean
    """
    N, D = X.shape
    T = Y.shape[1]

    jitter = JITTER
    def _unpack_params(params):
        return np.exp(params.reshape(M, D+T)[:, D:]), transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        A, Z = _unpack_params(params)

        if not train_A:
            A = np.ones_like(A) * default_variance

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)

        Kuu = np.diag(A[:, 0])
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        cov = Sigma + Qff + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)
        
        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

def dtc_independent_causal_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_A=False, default_variance=1.0, D=None):
    """
    function values at X are deterministic given u
    via the conditial mean

    here we ask that Sigma do all the work in explaining variance
    from this mean
    """
    if D is None:
        N, D = X.shape
    else:
        N = X.shape[1]
        D = D
        
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return np.exp(params.reshape(M, D+T)[:, D:]), transform.forward(params.reshape(M, D+T)[:, :D])

    def bound(params):
        var, Z = _unpack_params(params)

        if not train_A:
            var = np.ones_like(var) * default_variance

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)

        Qff = Kfu @ np.diag(var[:, 0]) @ Kfu.T

        cov = Sigma + Qff + np.eye(N) * jitter
        L = np.linalg.cholesky(cov)

        logdet = 2 * np.sum(np.log(np.diag(L)))
        A = np.linalg.solve(L, Y)
        quad = np.trace(A.T @ A)
        trace = 0
        return -0.5 * (N * T * np.log(2 * np.pi) + T * logdet + quad + trace).sum()
    return bound, _unpack_params

