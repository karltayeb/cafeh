import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .kls import unit_normal_kl, normal_kl, categorical_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class SpikeSlabSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores
    from .model_queries import get_credible_sets

    def __init__(self, X, Y, K, snp_ids, tissue_ids, prior_activity, prior_variance, tolerance=1e-6, **kwargs):
        """
        X [N x N] covariance matrix
            if X is [T x N x N] use seperate embedding for each tissue
        Y [T x N] matrix, Nans should be converted to 0s?
        """

        self.X = X
        self.Y = Y
        self.K = K
        self.T = Y.shape[0]
        self.N = Y.shape[1]

        self.prior_activity = prior_activity
        self.prior_variance = prior_variance * np.ones((self.T, self.K))

        self.snp_ids = snp_ids
        self.tissue_ids = tissue_ids

        # initialize variational parameters
        pi = np.random.random((self.N, self.K)) + 1
        pi = pi / pi.sum(0)

        self.pi = pi
        self.global_sign = np.ones(self.N)

        self.weights = np.random.random((self.T, self.K, self.N)) * 5
        #self.weights = np.ones((self.T, self.K))
        self.active = np.ones((self.T, self.K))
        #self.active = np.random.random((self.T, self.K))

        self.elbos = []
        self.tolerance = tolerance

    ################################
    # UPDATE AND FITTING FUNCTIONS #
    ################################
    def _flip(self, k, thresh=0.9):
        """
        flip snps and zscores to avoid having two blocks of negatively correlated snps
        k is the component to operate on
        thresh controls how far from the lead snp of the component to flip signs on

        the model keeps a global record of whats changed
        """
        sign = np.ones(self.N)
        lead_snp = self.pi[:, k].argmax()
        switch = (self.X[lead_snp] < 0) & (np.abs(self.X[lead_snp]) > thresh)
        sign[switch] *= -1

        # update data and ld matrices to relfect this change in sign
        self.Y = (self.Y * sign).astype(np.float64)
        if self.X.ndim == 2:
            self.X = self.X * np.outer(sign, sign).astype(np.float64)
        else:
            self.X = self.X * np.outer(sign, sign).astype(np.float64)[None]
        self.global_sign = self.global_sign * sign

    def _diffuse_pi(self, width, components=None, bizarro=False):
        """
        spread mass of pi around snps in high ld
        cutoff is width for edges to include in graph diffusion
        """
        if components is None:
            components = np.arange(self.K)
        if width < 1.0:
            X = self.X
            if bizarro:
                X = np.abs(X)
            if X.ndim == 3:
                X = np.mean(X, axis=0)
            transition = X * (X >= width).astype(np.float64)
            inv_degree = np.diag(1 / (transition.sum(1)))
            transition = inv_degree @ transition
            self.pi[:, components] = (transition.T @ self.pi)[:, components]

    def _compute_residual(self, k=None):
        """
        residual computation, works when X is 2d or 3d
        k is a component to exclude from residual computation
        """
        prediction = self._compute_prediction(k)
        residual = self.Y - prediction
        return residual

    def _compute_weight_vars(self):
        #weight_vars = (self.prior_variance * self.active)[..., None] * self.pi.T[None]
        weight_var = self.prior_variance
        weight_var = weight_var / (1 + weight_var)
        return weight_var[..., None] * np.ones_like(self.weights)

    def _compute_prediction(self, k=None):
        prediction = np.zeros_like(self.Y)
        if k is not None:
            prediction -= (self.X @ self.pi[:, k][:, None] * (self.weights[:, k] * self.active[:, k][:, None]).T).T
        for k in range(self.K):
            prediction += (self.X @ self.pi[:, k][:, None] * (self.weights[:, k] * self.active[:, k][:, None]).T).T
        return prediction

    def update_weights(self, components=None):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.K)

        old_weights = self.weights.copy()
        weight_var = self._compute_weight_vars()

        for k in components:
            r_k = self._compute_residual(k)
            self.weights[:, k, :] = r_k[:, :] * weight_var[:, k] * self.active[:, k][:, None]

        weight_diff = np.abs(old_weights - self.weights).max()
        return weight_diff

    def update_active(self, components=None):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.K)

        old_active = self.active.copy()
        weight_var = self._compute_weight_vars()
        for k in components:
            r_k = self._compute_residual(k)
            for t in range(self.T):
                # q(s = 0)
                off = np.log(1 - self.prior_activity[k])
                on = r_k[t] * self.weights[t, k] - 0.5 * (self.weights[t, k] ** 2 + weight_var[t, k])\
                    - normal_kl(self.weights[t, k], weight_var[t, k], 0, self.prior_variance[t, k])
                on = on @ self.pi[:, k] + np.log(self.prior_activity[k])
                self.active[t, k] = 1 / (1 + np.exp(-(on - off)))

        active_diff = np.abs(old_active - self.active).max()
        return active_diff

    def update_pi(self, components=None):
        """
        update pi
        """
        if components is None:
            components = np.arange(self.K)

        old_pi = self.pi.copy()
        active_components = (self.active.max(0) > 1e-2)
        weight_var = self._compute_weight_vars()
        #for k in components[active_components[components]]:
        for k in components:
            # compute residual
            r_k = self._compute_residual(k)

            pi_k = (r_k * self.weights[:, k]
                    - 0.5 * (self.weights[:, k] ** 2 + weight_var[:, k])
                    - normal_kl(self.weights[:, k], weight_var[:, k], 0, self.prior_variance[:, k][:, None] * np.ones_like(weight_var[:, k]))
                    )
            pi_k = pi_k.T @ self.active[:, k]

            # normalize to probabilities
            pi_k = np.exp(pi_k - pi_k.max())
            pi_k = pi_k / pi_k.sum()
            self.pi[:, k] = pi_k

        # component not active-- back to prior
        #for k in components[~active_components[components]]:
        #    self.pi[:, k] = np.ones(self.N) / self.N

        pi_diff = np.abs(self.pi - old_pi).max()
        return pi_diff

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
            components = np.arange(self.K)

        new_variances = np.zeros((self.T, components.size))
        
        """
        r = np.array([self._compute_residual(k=k) for k in range(self.K)])  # K x T x N
        for t in range(self.T):
            for i, k in enumerate(components):
                bf = lambda x: -np.power(x**2 + 1, -0.5) * np.sum(np.exp(np.power(r[k, t]**2 / 2 * x**2 / (1 + x**2), self.pi[:, k])))
                new_variances[t, i] = minimize_scalar(bf).x ** 2
        """
        for t in range(self.T):
            for i, k in enumerate(components):
                weight_var = self.prior_variance[t, k] / (1 + self.prior_variance[t, k])
                weight = self.weights[t, k]
                f = lambda x: normal_kl(weight, weight_var, 0, x**2)
                new_variances[t, i] = minimize_scalar(f).x ** 2
        
        """

        z = np.array([self._compute_residual(k=k) @ self.pi[:, k] for k in range(self.K)]).T
        for t in range(self.T):
            for i, k in enumerate(components):
                log_bf = lambda x: -1 *(-0.5 * np.log(x**2 + 1) + (z[t, k] ** 2) / 2 * (x**2 / (1 + x**2)))
                new_variances[t, i] = minimize_scalar(log_bf).x ** 2
        """

        self.prior_variance[:, components] = new_variances
        #self.prior_variance = temp.sum() / temp.size


    def _fit(self, max_outer_iter=1000, max_inner_iter=1, bound=False, verbose=False, components=None, diffuse=1.0, update_weights=True, update_active=True, update_pi=True, update_prior_variance=False, update_prior_activity=False):
        """
        loop through updates until convergence
        """
        if bound:
            self.elbos.append(self.compute_elbo())
        
        for i in range(max_outer_iter):
            # update weights and activities
            for _ in range(max_inner_iter):
                if update_prior_variance:
                    self.update_prior_variance(components)

                if update_weights:
                    diff1 = np.sum([self.update_weights(components) for _ in range(10)])
                else:
                    diff1 = 0.0

                if update_active:
                    diff2 = self.update_active(components)
                else:
                    diff2 = 0.0

                if diff1 < self.tolerance and diff2 < self.tolerance:
                    break
            
            # update pi
            for _ in range(max_inner_iter):
                if update_pi:
                    diff3 = np.sum([self.update_pi(components) for _ in range(10)])
                else:
                    diff3 = 0

                if diff3 < self.tolerance:
                    break
            self._diffuse_pi(diffuse)


            # after each inner loop record elbo
            if bound:
                self.elbos.append(self.compute_elbo())

            # check for convergence in ELBO
            if bound and np.abs(self.elbos[-1] - self.elbos[-3]) < self.tolerance:
                if verbose:
                    print('ELBO converged at iter {}'.format(i))
                break

            # check for convergance in varitional parameters
            elif diff1 < self.tolerance and diff2 < self.tolerance and diff3 < self.tolerance:
                if verbose:
                    print('Parameters converged at iter {}'.format(i))
                break

    def _forward_fit_step(self, l, max_inner_iter=1, max_outer_iter=1000,
                          diffuse=1.0, quantile=0.0,
                          bound=False, verbose=False, restarts=1, plots=False):
        """
        fit self as though there were only l components
        T initializations with unit weight at each tissue, pick best solution among them
        """
        K = self.K
        init_pi = self.pi.copy()
        init_active = self.active.copy()
        init_weights = self.weights.copy()

        restart_dict = {}
        elbos = []

        # initialize pi based on residuals
        residual = self._compute_residual()
        sq_err = np.max(residual**2, axis=0)
        pi = sq_err * (sq_err > np.quantile(sq_err, quantile))
        pi = residual.mean(0)
        pi = np.exp(pi - pi.max())
        pi = pi / pi.sum()

        self.pi[:, l-1] = pi
        #self._diffuse_pi(0.1, components=np.arange(l-1, l))

        if plots:
            plt.scatter(np.arange(pi.size), self.pi[:, l-1])
            plt.show()

        # positive initialization
        for i in range(restarts):
            self.pi = init_pi
            self.active = init_active
            self.weights = init_weights

            # initialize activity to random
            active_t = init_active.copy()
            active_t[:, l-1] = np.random.random(self.T)

            # initialize weights something random
            t = np.random.choice(self.T)
            weights_t = init_weights.copy()
            #weights_t[:, l-1] = np.random.normal(size=self.T)
            #weights_t[:, l-1] = np.eye(self.T)[t]
            weights_t[:, l-1] = np.zeros_like(self.T)

            # initialize pi of the lth component, weigh snps with poor predictions heavier
            pi_t = init_pi.copy()
            pi_t[:, l-1] = pi

            self.active = active_t
            self.weights = weights_t
            self.pi = pi_t

            #self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l-1, l), diffuse=diffuse)

            # fit the whole model up to this component
            # self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l-1, l), diffuse=diffuse)
            self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l), diffuse=diffuse)

            restart_dict[i] = (self.pi.copy(), self.active.copy(), self.weights.copy())
            elbos.append(self.compute_elbo())

        select = np.argmax(elbos)
        self.elbos.append(elbos[select])
        new_pi, new_active, new_weights = restart_dict[select]

        self.pi = new_pi
        self.active = new_active
        self.weights = new_weights

        return restart_dict, elbos

    def forward_fit(self, early_stop=False, max_inner_iter=1, max_outer_iter=1000,
                    diffuse=1.0, quantile=0.0, flip=False,
                    bound=False, verbose=False, restarts=1, plots=False):
        """
        forward selection scheme for variational optimization
        fit first l components with weights initialized to look at each tissue
        select the best solution (by elbo) among tissue initialization

        fit first l+1 components
        """
        self.weights = np.zeros_like(self.weights)
        self.active = np.zeros_like(self.active)

        for l in range(1, self.K+1):
            print('Forward fit, learning {} components'.format(l))
            self._forward_fit_step(
                l, max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter,
                diffuse=diffuse, quantile=quantile,
                bound=bound, verbose=verbose, restarts=restarts, plots=plots)

            # orient nearby snps and retrain component
            """
            if flip:
                self._flip(k=l-1, thresh=0.9)
                self._forward_fit_step(
                    l, max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter,
                    diffuse=diffuse, quantile=quantile,
                    bound=bound, verbose=verbose, restarts=restarts, plots=plots)
            """
            if plots:
                self.plot_components()

            # if the next step turned off the component, all future steps will
            # zero them out and do a final fit of the self
            # if self.pi[:, l-1].max() < 0.01 and early_stop:
            if early_stop and self.active[:, l-1].max() < 0.5:
                print('learned inactive components')
                # zero initialize the components
                self.active[:, l:] = 1 - self.prior_activity[l:]
                self.weights[:, l:] = 0
                break

        print('finalizing components')
        #self._fit(max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter, bound=bound, diffuse=diffuse, verbose=verbose)
        if plots:
            self.plot_components()

    def diffusion_fit(self, schedule):
        for i, rate in enumerate(schedule):
            self._fit(max_outer_iter=5, verbose=True)
            transition = np.abs(self.X) * (np.abs(self.X) > rate)
            degree = np.diag(1 / (transition.sum(1)))
            transition = degree @ transition
            self.pi = transition.T @ self.pi

    def compute_elbo(self, active=None):
        bound = 0 
        weight_var = self.prior_variance / (1 + self.prior_variance)

        if active is None:
            active = self.active

        """
        if self.X.ndim == 2:
            Kzz = self.pi.T @ self.X @ self.pi
            Kzz = Kzz + np.diag(np.ones(self.K) - np.diag(Kzz))
            self.X
        for t in range(self.T):
            if self.X.ndim == 3:
                Kzz = self.pi.T @ self.X[t] @ self.pi
                Kzz = Kzz + np.diag(np.ones(self.K) - np.diag(Kzz))

            bound += self.Y[t] @ (self.pi @ W[t])
            bound += -0.5 * W[t] @ Kzz @ W[t]
            bound += -0.5 * np.sum(
                (self.weights[t]**2 + weight_var[t]) * active[t] - (self.weights[t] * active[t])**2
            )
        """
        wv = self._compute_weight_vars()
        bound = np.sum([((self.Y[t] * self.weights[t] - 0.5 * (self.weights[0]**2 * wv[t]) ) * self.pi.T).sum(1) @ self.active[t] for t in range(self.T)])
        
        KL = 0
        for t in range(self.T):
            for k in range(self.K):
                # KL (q(w|s) || p(w | s))
                for j in range(self.N):
                    KL += normal_kl(self.weights[t, k, j], weight_var[t, k], 0, self.prior_variance[t, k]) * active[t, k] * self.pi[j, k]
                KL += normal_kl(0, self.prior_variance[t, k], 0, self.prior_variance[t, k]) * (1 - active[t, k])

                # KL (q(s) || p(s))
                KL += categorical_kl(
                    np.array([active[t, k], 1 - active[t, k]]),
                    np.array([self.prior_activity[k], 1 - self.prior_activity[k]])
                )

        for k in range(self.K):
            # KL (q(z) || p (z))
            KL += categorical_kl(self.pi[:, k], np.ones(self.N) / self.N)

        bound -= KL
        return bound 

    def sort_components(self):
        """
        reorder components so that components with largest weights come first
        """
        order = np.flip(np.argsort(np.abs(self.weights).max(0)))
        self.weights = self.weights[:, order]
        self.active = self.active[:, order]
        self.pi = self.pi[:, order]

    def save(self, output_dir, model_name):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

