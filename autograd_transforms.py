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

###############
#  transforms #
###############

def normalize(X):
    """
    scale a vector to be unit
    """
    return X / np.sqrt(np.sum(X**2, 1))[:, None]

def cumsign(x):
    """
    cumulative sign function-- keeps track of sign of cumulative product
    since we cant take log of negative values
    """
    neg = (x < 0)
    return -2 * (np.mod(np.cumsum(neg, axis=1), 2) - 0.5)

def cumprod(x):
    """
    write cumulative product in terms of sums since autograd supports that
    """
    return cumsign(x) * np.exp(np.cumsum(np.log(np.abs(x)), axis=1))

def logistic(X):
    """
    logistic function, squeeze [0, 1]
    """
    return 1 / (1 + np.exp(-X))

def inverse_logistic(Y):
    """
    inverse of the logistic function 1 / (1 + exp(-x))
    """
    return -1 * np.log(1 / Y - 1)

def spherical(X):
    """
    squeeze unconstrained vector
    into [0, pi] x .. x [0, pi] x [0, 2pi] box
    """
    bound = np.ones(X.shape[1] - 1) * np.pi
    bound[-1] = 2 * np.pi
    return bound * logistic(X[:, :-1])

def inverse_spherical(rho):
    """
    transform spherical coordinates to unconcstrained space
    """
    bound = np.ones(rho.shape[1]) * np.pi
    bound[-1] = 2 * np.pi
    return inverse_logistic(rho / bound)

def spherical_to_coords(rho):
    """
    return coordinates in R^n given
    spheirical coordinates
    """
    rho = np.atleast_2d(rho)
    sins = np.sin(rho)
    coss = np.cos(rho)
    cp = cumprod(sins)
    return np.hstack([np.ones((cp.shape[0], 1)), cp]) * np.hstack([coss, np.ones((coss.shape[0], 1))])

def coords_to_spherical(coords):
    """
    return spherical coordinates given coordinates in R^n
    """
    coords = np.atleast_2d(coords)

    rho0 = np.arccos(
        coords / np.flip(np.sqrt(np.cumsum(np.flip(coords ** 2, axis=1), axis=1)), axis=1)
    )[:, :-1]

    alt = 2 * np.pi - np.arccos(
        coords / np.flip(np.sqrt(np.cumsum(np.flip(coords ** 2, axis=1), axis=1)), axis=1)
    )[:, -2]

    if np.any(coords[:, -1] < 0):
        rho0[coords[:, -1] < 0, -1] = alt[coords[:, -1] < 0]
    return rho0


class identity_transform:
    @staticmethod
    def forward(Z_unconstrained):
        return Z_unconstrained

    @staticmethod
    def backward(Z_transformed):
        return Z_transformed

class spherical_transform:
    @staticmethod
    def forward(Z_unconstrained):
        return spherical_to_coords(spherical(Z_unconstrained))

    @staticmethod
    def backward(Z_transformed):
        return np.hstack([
            inverse_spherical(coords_to_spherical(Z_transformed)),
            np.ones((Z_transformed.shape[0], 1))])

class normalize_transform:
    @staticmethod
    def forward(Z_unconstrained):
        return normalize(Z_unconstrained)

    @staticmethod
    def backward(Z_transformed):
    	# normalizing a unit vector will give a unit vector
        return Z_transformed
