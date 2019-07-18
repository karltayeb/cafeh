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

def joint_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    to maximize joint likelihood of data and inducing points
    (u, z) are free parameters

    we also include function that puts in the optimal u for a given z
    so that we only need to do gradients on z
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)
        
        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + Kff - Qff + np.eye(mean.shape[0]) * jitter

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior

    def log_gp_conditional_optimal_v(params):
        _, Z = _unpack_params(params)
        
        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        # whitened mean u = Lv --> v
        # KufKuu^{-1}u = KufLuu^{-T}v
        v = update_v(params)
        mean = LinvKuf.T @ v
        cov = Sigma + Kff - Qff

        L = np.linalg.cholesky(cov + np.eye(N) * jitter)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean), 0)    
        likelihood = np.sum(likelihood - 0.5 * logdet)

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v), 0)
        prior = np.sum(prior - 0.5 * prior_logdet)
        return likelihood + prior

    def update_v(params):
        _, Z = _unpack_params(params)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LuuinvKuf = np.linalg.solve(Luu, Kfu.T)
        
        Qff = LuuinvKuf.T @ LuuinvKuf
        cov = Sigma + Kff - Qff
        L = np.linalg.cholesky(cov + np.eye(N) * jitter)  
        LuuinvKufLinv = np.linalg.solve(L, LuuinvKuf.T).T
        return np.linalg.solve(np.eye(M) + LuuinvKuf @ LuuinvKuf.T, LuuinvKufLinv @ Y)
    
    return joint, _unpack_params

def joint_fitc_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    optimize joint with FITC assumption
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)

        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + np.diag(np.diag(Kff - Qff)) + np.eye(mean.shape[0]) * jitter

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior
    
    return joint, _unpack_params
    
def joint_independent_causal_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    c is iid causal effect size
    u = Kuu c

    use full conditional
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)
        
        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Kuu = Kuu @ np.eye(M) @ Kuu
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + (Kff - Qff) + np.eye(mean.shape[0]) * 2 * np.min(np.abs(np.diag(Kff - Qff)))

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior
    
    return joint, _unpack_params


def joint_independent_causal_fitc_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    c is iid causal effect size
    u = Kuu c

    assume FITC
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)

        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = kernel(Z, Z)
        Kuu = Kuu @ np.eye(M) @ Kuu
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + np.diag(np.abs(np.diag(Kff - Qff))) + np.eye(mean.shape[0]) * jitter

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior  
    return joint, _unpack_params

def joint_independent_inducing_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    u are iid
    use full conditional
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)
        
        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = np.eye(M)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + Kff - Qff + np.eye(mean.shape[0]) * jitter

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior
    
    return joint, _unpack_params


def joint_independent_inducing_fitc_functions(X, Y, Sigma, kernel=linear_kernel, M=5, transform=identity_transform, train_v=False):
    """
    u are iid
    use full conditional
    """
    N, D = X.shape
    T = Y.shape[1]
    
    jitter = JITTER
    def _unpack_params(params):
        return params.reshape(M, D+T)[:, D:], transform.forward(params.reshape(M, D+T)[:, :D])

    def joint(params):
        v, Z = _unpack_params(params)

        if not train_v:
            v = np.ones_like(v)

        # use a linear kernel for now
        Kff = kernel(X, X)
        Kfu = kernel(X, Z)
        Kuu = np.eye(M)
        Luu = np.linalg.cholesky(Kuu + np.eye(M) * jitter)

        LinvKuf = np.linalg.solve(Luu, Kfu.T)
        Qff = LinvKuf.T @ LinvKuf

        mean = LinvKuf.T @ v
        cov = Sigma + np.diag(np.abs(np.diag(Kff - Qff))) + np.eye(N) * jitter

        L = np.linalg.cholesky(cov)
        Linv_mean = np.linalg.solve(L, (Y - mean))

        logdet = 2 * np.sum(np.log(np.diag(L)))
        likelihood = np.sum(norm.logpdf(Linv_mean))    
        likelihood = likelihood - 0.5 * Y.shape[1] * logdet

        prior_logdet = 2 * np.sum(np.log(np.diag(Luu)))
        prior = np.sum(norm.logpdf(v))
        prior = prior - 0.5 * v.shape[1] * prior_logdet
        return likelihood + prior
    
    return joint, _unpack_params