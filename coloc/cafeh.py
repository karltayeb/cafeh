import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar
from .utils import np_cache_class, gamma_logpdf
from functools import lru_cache
import time

class CAFEH:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip, check_convergence, get_expected_weights

    def __init__(self, X, Y, K, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, tolerance=1e-5):
        """
        X [N x N] covariance matrix
            if X is [T x N x N] use seperate embedding for each tissue
        Y [T x N] matrix, Nans should be converted to 0s?
        """

        # precompute svd, need for computing elbo
        T, N = Y.shape

        self.X = X
        self.Y = Y
        self.dims = {'N': N, 'T': T, 'K': K}

        # priors
        self.prior_precision = np.ones((T, K)) * prior_variance
        self.prior_component_precision = np.ones(K)
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi  if (prior_pi is not None) else np.ones(N) / N

        # ids
        self.tissue_ids = tissue_ids if (tissue_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)

        # initialize variational parameters
        self.pi = np.ones((K, N)) / N
        self.weight_means = np.zeros((T, K, N))

        prior_variance = 1 / (self.prior_precision * self.prior_component_precision)
        self.weight_vars = (prior_variance / (prior_variance + 1))[:, :, None] * np.ones((T, K, N))
        self.active = np.ones((T, K))

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0

        self.alpha0 = 1.0
        self.beta0 = 1e-10

        self.alpha0_component = 1.0
        self.beta0_component = 1.0

    ################################
    # UPDATE AND FITTING FUNCTIONS #
    ################################
    def prior_variance(self):
        """
        return prior variance
        """
        return 1 / (self.prior_precision * self.prior_component_precision)

    @np_cache_class()
    def _compute_prediction_component(self, active, pi, weights):
        if np.ndim(self.X) == 2:
            return active[:, None] * (pi * weights) @ self.X
        else:
            return active[:, None] * np.einsum(
                'tn, tnm->tm', (pi * weights), self.X)

    def compute_prediction_component(self, k):
        active= self.active[:, k]
        pi = self.pi[k]
        weights = self.weight_means[:, k]
        return self._compute_prediction_component(active, pi, weights)

    def compute_prediction(self, k=None):
        prediction = np.zeros_like(self.Y)
        for l in range(self.dims['K']):
            prediction += self.compute_prediction_component(l)
        if k is not None:
            prediction -= self.compute_prediction_component(k)
        return prediction

    def compute_residual(self, k=None):
        """
        residual computation, works when X is 2d or 3d
        k is a component to exclude from residual computation
        """
        prediction = self.compute_prediction(k)
        residual = self.Y - prediction
        return residual

    def _update_weight_component(self, k, ARD=False):
        r_k = self.compute_residual(k)
        if ARD:
            second_moment = (self.weight_vars[:, k] + self.weight_means[:, k] **2) @ self.pi[k]
            alpha = self.alpha0 + 0.5
            beta = self.beta0 + second_moment / 2 * self.prior_component_precision[k]
            self.prior_precision[:, k] = np.clip((alpha - 1) / beta, 1e-10, 1e5)

            #alpha = self.alpha0_component + self.dims['T'] / 2
            #beta = np.sum(second_moment / 2 * self.prior_precision[:, k]) + self.beta0_component
            #self.prior_component_precision[k] = np.clip((alpha - 1) / beta, 1e-5, 1e10)

            #rebalance
            #z = np.log10(self.prior_component_precision[k])
            #self.prior_precision[:, k] = self.prior_entry_precision[:, k] * np.power(10.0, z)
            #self.prior_component_precision[k] /= np.power(10.0, z)

        for tissue in range(self.dims['T']):
            precision = np.diag(self.get_ld(tissue=tissue)) + (1 / self.prior_variance()[tissue, k])
            variance = 1 / precision
            mean = r_k[tissue] * variance

            self.weight_vars[tissue, k] = variance
            self.weight_means[tissue, k] = mean

    def update_weights(self, components=None, ARD=False):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_weight_component(k, ARD=ARD)

    def _update_active_component_tissue(self, r_k, tissue, component):
        off = np.log(1 - self.prior_activity[component] + 1e-10)
        p = (self.pi[component] * self.weight_means[tissue, component])
        tmp1 = r_k[tissue] @ p
        tmp2  = -0.5 * (self.weight_means[tissue, component]**2 + self.weight_vars[tissue, component]) @ self.pi[component]
        tmp3 = -1 * normal_kl(
            self.weight_means[tissue, component], self.weight_vars[tissue, component],
            0, self.prior_variance[tissue, component]) @ self.pi[component]
        on = tmp1 + tmp2 + tmp3 + np.log(self.prior_activity[component] + 1e-10)
        self.active[tissue, component] = 1 / (1 + np.exp(-(on - off)))

    def _update_active_component(self, k, ARD=False):
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_activity[k] = (a - 1) / (a + b - 2)
        r_k = self.compute_residual(k)
        for t in range(self.dims['T']):
            self._update_active_component_tissue(r_k, t, k)

    def _update_active_component2(self, k, ARD=False):
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_activity[k] = (a - 1) / (a + b - 2)
        r_k = self.compute_residual(k)

        off = np.log(1 - self.prior_activity[component] + 1e-10)
        p = (self.pi[component][:, None] * self.weight_means[:, component])
        tmp1 = r_k @ p
        tmp2  = -0.5 * (self.weight_means[:, component]**2 + self.weight_vars[:, component]) @ self.pi[component]
        tmp3 = -1 * normal_kl(
            self.weight_means[:, component], self.weight_vars[:, component],
            0, self.prior_variance[:, component]) @ self.pi[component]
        on = tmp1 + tmp2 + tmp3 + np.log(self.prior_activity[component] + 1e-10)
        self.active[:, component] = 1 / (1 + np.exp(-(on - off)))

    def update_active(self, components=None, ARD=False):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_active_component(k, ARD=ARD)

    def update_active2(self, components=None, ARD=False):
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_active_component(k, ARD=ARD)

    def _update_pi_component(self, k, ARD=False):
        # compute residual
        r_k = self.compute_residual(k)

        pi_k = (r_k * self.weight_means[:, k]
                - 0.5 * (self.weight_means[:, k] ** 2 + self.weight_vars[:, k]) * self.get_diag()
                - normal_kl(
                    self.weight_means[:, k], self.weight_vars[:, k],
                    0, self.prior_variance()[:, k][:, None] * np.ones_like(self.weight_vars[:, k]))
                )
        pi_k = pi_k.T @ self.active[:, k]

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()
        self.pi.T[:, k] = pi_k

    def update_pi(self, components=None):
        """
        update pi
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_pi_component(k)

    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        """
        loop through updates until convergence
        """
        init_time = time.time()

        if components is None:
            components = np.arange(self.dims['K'])

        for i in range(max_iter):
            for l in components:
                if update_weights: self._update_weight_component(l, ARD=ARD_weights)        
                if update_pi: self._update_pi_component(l)
                if update_active: self._update_active_component(l, ARD=ARD_active)

            self.elbos.append(self.compute_elbo())
            if verbose: print("Iter {}: {}".format(i, self.elbos[-1]))

            cur_time = time.time()
            if self.check_convergence():
                if verbose:
                    print('ELBO converged with tolerance {} at iter: {}'.format(self.tolerance, i))
                break

        self.run_time += cur_time - init_time
        if verbose:
            print('cumulative run time: {}'.format(self.run_time))

    def forward_fit(self, max_iter=1000, verbose=False, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        for k in range(self.dims['K']):
            self.fit(max_iter, verbose, np.arange(k), update_weights, update_active, update_pi, ARD_weights, ARD_active)

    def compute_elbo(self, active=None):
        bound = 0 
        if active is None:
            active = self.active

        expected_conditional = 0
        KL = 0

        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        for tissue in range(self.dims['T']):
            p = self.active[tissue] @ (self.pi * self.weight_means[tissue])
            expected_conditional += np.inner(self.Y[tissue], p)

            z = self.pi * self.weight_means[tissue] #* self.active[tissue]
            z = z @ self.get_ld(tissue=tissue) @ z.T
            z = z - np.diag(np.diag(z))
            expected_conditional += -0.5 * z.sum()
            expected_conditional += -0.5 * ((self.weight_means[tissue] ** 2 + self.weight_vars[tissue]) * self.pi).sum()

        # KL(q(W | S) || p(W)) = KL(q(W | S = 1) || p(W)) q(S = 1) + KL(p(W) || p(W)) (1 - q(S = 1))
        KL += np.sum(
            normal_kl(self.weight_means, self.weight_vars, 0, self.prior_variance()[..., None])
            * (self.active[..., None] * self.pi[None])
        )

        KL += np.sum(bernoulli_kl(self.active, self.prior_activity[None]))
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )
        # TODO ADD lnp(prior_weight_variance) + lnp(prior_slab_weights)
        # expected_conditional += gamma_logpdf(self.prior_component_precision, self.alpha0_component, self.beta0_component).sum()
        expected_conditional += gamma_logpdf(self.prior_precision, self.alpha0, self.beta0).sum()
        return expected_conditional - KL

    def get_ld(self, tissue=None, snps=None):
        """
        get ld matrix
        this function gives a common interface to
        (tisse, snp, snp) and (snp, snp) ld
        """
        if np.ndim(self.X) == 2:
            tissue = None
        return np.squeeze(self.X[tissue][..., snps, :][..., snps])

    def get_diag(self):
        return np.atleast_2d(np.squeeze(
            np.array([np.diag(X) for X in np.atleast_3d(self.X)])))

    def sort_components(self):
        """
        reorder components so that components with largest weights come first
        """
        order = np.flip(np.argsort(np.abs(self.get_expected_weights()).max(0)))
        self.weight_means = self.weight_means[:, order]
        self.active = self.active[:, order]
        self.pi = self.pi[order]
        self._compute_prediction_component.cache_clear()

    def save(self, output_dir, model_name, save_data=False):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]

        if not save_data:
            X = self.__dict__.pop('X')
            Y = self.__dict__.pop('Y')
            
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

        if not save_data:
            self.__dict__['X'] = X
            self.__dict__['Y'] = Y

