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

def cumsign(x, axis=None):
    """
    cumulative sign function-- keeps track of sign of cumulative product
    since we cant take log of negative values
    """
    neg = (x < 0)
    return -2 * (np.mod(np.cumsum(neg, axis=axis), 2) - 0.5)

def cumprod(x, axis=None):
    """
    write cumulative product in terms of sums since autograd supports that
    """
    return cumsign(x, axis=axis) * np.exp(np.cumsum(np.log(np.abs(x)), axis=axis))

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

def spherical(x):
    """
    squeeze unconstrained vector
    into [0, pi] x .. x [0, pi] x [0, 2pi] box
    """
    x = x[:, :-1]
    bound = np.ones(x.shape[1]) * np.pi
    bound[-1] = 2 * np.pi
    return bound * logistic(x)

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
    sines = np.sin(rho)
    cosines = np.cos(rho)
    sine_products = cumprod(sines, axis=1)
    return np.hstack([np.ones((sine_products.shape[0], 1)), sine_products]) * np.hstack([cosines, np.ones((cosines.shape[0], 1))])

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

class hemisspherical_transform:
    @staticmethod
    def forward(x):
        x = logistic(x) * np.pi
        rho = np.atleast_2d(x)  # M x D-1
        sines = np.sin(rho)  # M x D-1
        cosines = np.cos(rho)
        sine_products = cumprod(sines, axis=1)

        y = np.hstack([np.ones([sine_products.shape[0], 1]), sine_products])[:, :-1] * cosines
        yn = np.sqrt(np.abs(1.0 - np.sum(y**2, axis=1)))[:, None]
        return np.hstack([y, yn])  # M x D

    @staticmethod
    def backward(y):
        square_sums = np.flip(np.cumsum(np.flip(y**2, axis=1), axis=1), axis=1)[:, :-1]
        arc_cosines = np.arccos(y[:, :-1] / np.sqrt(square_sums))
        arc_cosines = np.clip(arc_cosines, 1e-10, np.pi-1e-10)
        return inverse_logistic(arc_cosines / np.pi)

class spherical_transform2:
    @staticmethod
    def forward(x):
        bound = np.ones(x.shape[1]) * np.pi
        bound[-1] = 2 * np.pi
        x = logistic(x) * bound

        rho = np.atleast_2d(x)  # M x D-1
        sines = np.sin(rho)  # M x D-1
        cosines = np.cos(rho)
        sine_products = cumprod(sines, axis=1)
        return np.hstack([np.ones((sine_products.shape[0], 1)), sine_products]) * np.hstack([cosines, np.ones((cosines.shape[0], 1))])

    @staticmethod
    def backward(y):
        bound = np.ones(y.shape[1] - 1) * np.pi
        bound[-1] = 2 * np.pi

        square_sums = np.flip(np.cumsum(np.flip(y**2, axis=1), axis=1), axis=1)[:, :-1]
        arc_cosines = np.arccos(y[:, :-1] / np.sqrt(square_sums))
        arc_cosines = np.clip(arc_cosines, 1e-10, np.pi-1e-10)

        for i in range(y.shape[0]):
            if y[0, -1] < 0:
                arc_cosines[0, -1] = 2 * np.pi - arc_cosines[i, -1]
        return inverse_logistic(arc_cosines / bound)

class normalize_transform:
    @staticmethod
    def forward(Z_unconstrained):
        return normalize(Z_unconstrained)

    @staticmethod
    def backward(Z_transformed):
        # normalizing a unit vector will give a unit vector
        return Z_transformed
