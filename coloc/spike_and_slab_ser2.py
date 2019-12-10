import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class MVNFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip

    def __init__(self, X, Y, K, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, tolerance=1e-6):
        """
        X [N x N] covariance matrix
            if X is [T x N x N] use seperate embedding for each tissue
        Y [T x N] matrix, Nans should be converted to 0s?
        """

        # precompute svd, need for computing elbo
        u, s, vh = np.linalg.svd(X)
        T, N = Y.shape

        self.X = X
        self.U = u * np.sqrt(s)
        self.Y = Y

        self.dims = {'N': N, 'T': T, 'K': K}

        # priors
        self.prior_variance = np.ones((T, K)) * prior_variance
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi or np.ones(N) / N

        # ids
        self.tissue_ids = tissue_ids or np.arange(T)
        self.snp_ids = snp_ids or np.arange(N)

        # initialize variational parameters
        self.pi = np.ones((K, N)) / N
        self.weight_means = np.zeros((T, K, N))
        self.weight_vars = (self.prior_variance / (self.prior_variance + 1))[:, :, None] * np.ones((T, K, N))
        self.active = np.ones((T, K))

        self.elbos = []
        self.tolerance = tolerance

    ################################
    # UPDATE AND FITTING FUNCTIONS #
    ################################

    def _diffuse_pi(self, width, components=None, bizarro=False):
        """
        spread mass of pi around snps in high ld
        cutoff is width for edges to include in graph diffusion
        """
        if components is None:
            components = np.arange(self.dims['K'])
        if width < 1.0:
            X = self.X
            if bizarro:
                X = np.abs(X)
            if X.ndim == 3:
                X = np.mean(X, axis=0)
            transition = X * (X >= width).astype(np.float64)
            inv_degree = np.diag(1 / (transition.sum(1)))
            transition = inv_degree @ transition
            self.pi.T[:, components] = (transition.T @ self.pi.T)[:, components]

    def _compute_prediction_old(self, k=None):
        prediction = np.zeros_like(self.Y)
        if k is not None:
            prediction -= (self.X @ self.pi.T[:, k][:, None] * (self.weight_means[:, k] * self.active[:, k][:, None]).T).T
        for k in range(self.dims['K']):
            prediction += (self.X @ self.pi.T[:, k][:, None] * (self.weight_means[:, k] * self.active[:, k][:, None]).T).T
        return prediction

    def _compute_prediction(self, k=None):
        """
        compute expected prediction
        """
        prediction = np.zeros_like(self.Y)
        for tissue in range(self.dims['T']):
            prediction[tissue] = self.active[tissue] @ ((self.pi * self.weight_means[tissue]) @ self.X)
            if k is not None:
                prediction[tissue] -= self.active[tissue, k] * (self.pi[k] * self.weight_means[tissue, k]) @ self.X
        return prediction

    def _compute_residual(self, k=None):
        """
        residual computation, works when X is 2d or 3d
        k is a component to exclude from residual computation
        """
        prediction = self._compute_prediction(k)
        residual = self.Y - prediction
        return residual

    def _compute_moments(self, tissue, component):
        """
        first and second moment of tissue, component prediction
        E[(x^T z w s)], E[(x^T z w s)^2]
        """
        mu2 = self.active[tissue, component] * (self.pi[component]
               * (self.weight_means[tissue, component]**2
                  + self.weight_vars[tissue, component])) @ self.U**2
        mu = self.active[tissue, component] * (self.pi[component]
              * self.weight_means[tissue, component]) @ self.U
        return mu, mu2

    def _update_weight_component(self, k, ARD=False):
        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            if ARD:
                self.prior_variance[tissue, k] = np.inner(
                    (self.weight_vars[tissue, k] + self.weight_means[tissue, k]**2), self.pi[k]) \
                * self.active[tissue, k]

            precision = 1 + (1 / self.prior_variance[tissue, k])
            variance = 1 / precision
            mean = r_k[tissue] * variance

            self.weight_vars[tissue, k] = variance * np.ones(self.dims['N'])
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
        r_k = self._compute_residual(k)
        for t in range(self.dims['T']):
            self._update_active_component_tissue(r_k, t, k)

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

    def _update_pi_component(self, k, ARD=False):
        # compute residual
        r_k = self._compute_residual(k)

        pi_k = (r_k * self.weight_means[:, k]
                - 0.5 * (self.weight_means[:, k] ** 2 + self.weight_vars[:, k])
                - normal_kl(self.weight_means[:, k], self.weight_vars[:, k], 0, self.prior_variance[:, k][:, None] * np.ones_like(self.weight_vars[:, k]))
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
        if components is None:
            components = np.arange(self.dims['K'])

        self.elbos.append(self.compute_elbo())
        for i in range(max_iter):
            for l in components:
                if update_weights:
                    self._update_weight_component(l, ARD=ARD_weights)        
                
                if update_pi:
                    self._update_pi_component(l)
                
                if update_active:
                    self._update_active_component(l, ARD=ARD_active)

            self.elbos.append(self.compute_elbo())

            diff = self.elbos[-1] - self.elbos[-2]
            if (np.abs(diff) < self.tolerance):
                if verbose:
                    print('ELBO converged with tolerance {} at iter: {}'.format(self.tolerance, i))
                break

    def forward_fit(self, max_iter=1000, verbose=False, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        for k in range(self.dims['K']):
            self.fit(max_iter, verbose, np.arange(k), update_weights, update_active, update_pi, ARD_weights, ARD_active)

    def compute_elbo(self, active=None):
        bound = 0 
        if active is None:
            active = self.active

        """
        if self.X.ndim == 2:
            Kzz = self.pi.T.T @ self.X @ self.pi.T
            Kzz = Kzz + np.diag(np.ones(self.dims['K']) - np.diag(Kzz))
            self.X
        for t in range(self.dims['T']):
            if self.X.ndim == 3:
                Kzz = self.pi.T.T @ self.X[t] @ self.pi.T
                Kzz = Kzz + np.diag(np.ones(self.dims['K']) - np.diag(Kzz))

            bound += self.Y[t] @ (self.pi.T @ W[t])
            bound += -0.5 * W[t] @ Kzz @ W[t]
            bound += -0.5 * np.sum(
                (self.weight_means[t]**2 + self.weight_vars[t]) * active[t] - (self.weight_means[t] * active[t])**2
            )
        """
        expected_conditional = 0
        KL = 0

        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        for tissue in range(self.dims['T']):
            p = self.active[tissue] @ (self.pi * self.weight_means[tissue])
            expected_conditional += np.inner(self.Y[tissue], p)
            expected_conditional += -0.5 * (p @ self.X @ p)
            # add trace term
            for component in range(self.dims['K']):
                mu, mu2 = self._compute_moments(tissue, component)
                expected_conditional += -0.5 * np.sum(mu2 - mu**2)

        # KL(q(W | S) || p(W)) = KL(q(W | S = 1) || p(W)) q(S = 1) + KL(p(W) || p(W)) (1 - q(S = 1))
        KL += np.sum(
            normal_kl(self.weight_means, self.weight_vars, 0, self.prior_variance[..., None])
            * (self.active[..., None] * self.pi[None])
        )

        KL += np.sum(bernoulli_kl(self.active, self.prior_activity[None]))
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )

        # TODO ADD lnp(prior_weight_variance) + lnp(prior_slab_weights)
        return expected_conditional - KL

    def sort_components(self):
        """
        reorder components so that components with largest weights come first
        """
        order = np.flip(np.argsort(np.abs(self.weight_means).max(0)))
        self.weight_means = self.weight_means[:, order]
        self.active = self.active[:, order]
        self.pi.T = self.pi.T[:, order]

    def save(self, output_dir, model_name):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

