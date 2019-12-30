import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .kls import unit_normal_kl, normal_kl, categorical_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class SimpleMVNFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip

    def __init__(self, X, Y, K, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, tolerance=1e-3):
        """
        X [N x N] covariance matrix
            if X is [T x N x N] use seperate embedding for each tissue
        Y [T x N] matrix, Nans should be converted to 0s?
        """

        self.X = X
        self.Y = Y

        T, N = Y.shape
        self.dims = {'N': N, 'T': T, 'K': K}

        # priors
        self.prior_variance = np.ones((T, K)) * prior_variance
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi if (prior_pi is not None) else np.ones(N) / N

        # ids
        self.tissue_ids = tissue_ids if (tissue_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)

        # initialize variational parameters
        self.pi = np.ones((K, N)) / N
        self.weight_means = np.zeros((T, K))
        self.weight_vars = np.ones((T, K))
        self.active = np.ones((T, K))

        self.elbos = []
        self.tolerance = tolerance

        self.x_is_ld = True

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

    def _compute_residual(self, k=None):
        """
        residual computation, works when X is 2d or 3d
        k is a component to exclude from residual computation
        """
        prediction = self._compute_prediction(k)
        residual = self.Y - prediction
        return residual

    def _compute_prediction(self, k=None):
        W = self.weight_means * self.active
        if self.X.ndim == 2:
            prediction = W @ (self.X @ self.pi.T).T
            if k is not None:
                prediction -= W[:, k][:, None] * (self.X @ self.pi.T[:, k])[None]
        if self.X.ndim == 3:
            prediction = np.stack([W[t] @ (self.X[t] @ self.pi.T).T for t in range(self.dims['T'])])
            if k is not None:
                prediction -= np.stack(
                    [W[t, k] * (self.X[t] @ self.pi.T[:, k]) for t in range(self.dims['T'])]
                )
        return prediction

    def _compute_weight_var(self):
        self.weight_vars = self.prior_variance #* self.active
        self.weight_vars = self.weight_vars / (1 + self.weight_vars)
        return self.weight_vars

    def _update_weight_component(self, k, precomputed_residual=None, ARD=False):
        if ARD:
            self.prior_variance[:, k] = (self.weight_means[:, k]**2 + self.weight_vars[:, k]) * self.active[:, k]

        # get expected weights
        W = self.weight_means * self.active

        # compute residual
        r_k = precomputed_residual if (precomputed_residual is not None) else self._compute_residual(k)

        # update p(w | s = 1)
        self.weight_vars[:, k] = self.prior_variance[:, k] / (1 + self.prior_variance[:, k])
        self.weight_means[:, k] = self.weight_vars[:, k] * (r_k @ self.pi.T[:, k])

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
            # get expected weights
            self._update_weight_component(k, ARD)

    def _update_active_component(self, k, precomputed_residual=None, ARD=False):
        if ARD:
            pass

        r_k = precomputed_residual if (precomputed_residual is not None) else self._compute_residual(k)
        for t in range(self.dims['T']):
            off = np.log(1 - self.prior_activity[k] + 1e-10)
            on = r_k[t] @ self.pi.T[:, k] * self.weight_means[t, k] - 0.5 * (self.weight_means[t, k]**2 + self.weight_vars[t, k]) \
                - normal_kl(self.weight_means[t, k], self.weight_vars[t, k], 0, self.prior_variance[t, k]) \
                + np.log(self.prior_activity[k])

            self.active[t, k] = 1 / (1 + np.exp(-(on - off)))

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
            self._update_active_component(k, ARD)

    def _update_pi_component(self, k, precomputed_residual=None):
        # compute residual
        r_k = precomputed_residual if (precomputed_residual is not None) else self._compute_residual(k)
        W = self.active * self.weight_means

        # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
        pi_k = r_k * W[:, k][:, None]
        pi_k = pi_k.sum(0)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max() + 5)
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

    def update_prior_activity(self, components=None):
        """
        EB for prior activity
        """
        pass

    def update_prior_variance(self, components=None):
        """
        EB for prior variances-- this is crucial to getting sensible credible intervals at end
        """
        if components is None:
            components = np.arange(self.dims['K'])

        new_variances = np.zeros((self.dims['T'], components.size))
        
        """
        r = np.array([self._compute_residual(k=k) for k in range(self.dims['K'])])  # K x T x N
        for t in range(self.dims['T']):
            for i, k in enumerate(components):
                bf = lambda x: -np.power(x**2 + 1, -0.5) * np.sum(np.exp(np.power(r[k, t]**2 / 2 * x**2 / (1 + x**2), self.pi.T[:, k])))
                new_variances[t, i] = minimize_scalar(bf).x ** 2
        """
        for t in range(self.dims['T']):
            for i, k in enumerate(components):
                self.weight_vars = self._compute_weight_var()
                weight = self.weight_means[t, k]
                f = lambda x: normal_kl(weight, self.weight_vars, 0, x**2)
                new_variances[t, i] = minimize_scalar(f).x ** 2
        
        """

        z = np.array([self._compute_residual(k=k) @ self.pi.T[:, k] for k in range(self.dims['K'])]).T
        for t in range(self.dims['T']):
            for i, k in enumerate(components):
                log_bf = lambda x: -1 *(-0.5 * np.log(x**2 + 1) + (z[t, k] ** 2) / 2 * (x**2 / (1 + x**2)))
                new_variances[t, i] = minimize_scalar(log_bf).x ** 2
        """

        self.prior_variance[:, components] = new_variances
        #self.prior_variance = temp.sum() / temp.size


    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        """
        loop through updates until convergence
        """
        if components is None:
            components = np.arange(self.dims['K'])

        self.elbos.append(self.compute_elbo())
        for i in range(max_iter):
            for l in components:
                r_l = self._compute_residual(l)

                if update_weights:
                    self._update_weight_component(l, precomputed_residual=r_l, ARD=ARD_weights)        
                
                if update_pi:
                    self._update_pi_component(l, precomputed_residual=r_l)
                
                if update_active:
                    self._update_active_component(l, precomputed_residual=r_l, ARD=ARD_active)

            self.elbos.append(self.compute_elbo())
            
            if i % 5 == 0:
                if verbose:
                    print('Iter {}: {}'.format(i, self.elbos[-1]))

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
        W = self.weight_means * self.active
        self.weight_vars = self._compute_weight_var()

        if active is None:
            active = self.active

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

        KL = 0
        for t in range(self.dims['T']):
            for k in range(self.dims['K']):

                # KL (q(w|s) || p(w | s))
                KL += normal_kl(self.weight_means[t, k], self.weight_vars[t, k], 0, self.prior_variance[t, k]) * active[t, k]
                KL += normal_kl(0, self.prior_variance[t, k], 0, self.prior_variance[t, k]) * (1 - active[t, k])

                # KL (q(s) || p(s))
                KL += categorical_kl(
                    np.array([active[t, k], 1 - active[t, k]]),
                    np.array([self.prior_activity[k], 1 - self.prior_activity[k]])
                )

        for k in range(self.dims['K']):
            # KL (q(z) || p (z))
            KL += categorical_kl(self.pi.T[:, k], np.ones(self.dims['N']) / self.dims['N'])
        bound -= KL
        return bound 

    def sort_components(self):
        """
        reorder components so that components with largest weights come first
        """
        order = np.flip(np.argsort(np.abs(self.weight_means).max(0)))
        self.weight_means = self.weight_means[:, order]
        self.weight_vars = self.weight_vars[:, order]
        self.active = self.active[:, order]
        self.pi = self.pi[order]

    def save(self, output_dir, model_name):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

