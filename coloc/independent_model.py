import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class SpikeSlabSer:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores
    from .model_queries import get_credible_sets

    def __init__(self, X, Y, K, prior_weight_variance, prior_slab_weights, pi_prior):
        """
        Y [T x M] expresion for tissue, individual
        X [N x M] genotype for snp, individual
            potentially [T x N x M] if tissue specific correction

        prior_weight_variance [T, K]
            prior variance for weight of (tissue, component) loading
        prior_slab_weights [K]
            prior probability of sapling from slab in component k
        pi_prior: prior for multinomial,
            probability of sampling a snps as the active feature
        """

        # set data
        self.X = X
        self.Y = Y

        # set priors
        T, M = Y.shape
        N = X.shape[0]
        self.dims = {'N': N, 'M': M, 'T': T, 'K': K}

        self.prior_weight_variance = prior_weight_variance
        self.prior_slab_weights = prior_slab_weights
        self.pi_prior = pi_prior

        # initialize latent vars
        self.active = np.ones((T, K))
        self.weight_means = np.ones((T, K))
        self.weight_vars = np.ones((T, K))
        self.tissue_variance = np.ones(T)
        self.pi = np.ones((K, N)) / N

        # hyper-parameters
        self.alpha = 1e-10
        self.beta = 1e-10

        self.elbos = []
        self.tolerance = 1e-5

    def _compute_prediction(self, k=None):
        """
        compute expected prediction
        E[XB]
        """
        W = self.weight_means * self.active
        if self.X.ndim == 2:
            # [T, K] @ ([M, N] @ [N, K]).T = [T, M]
            prediction = W @ self.pi @ self.X
            if k is not None:
                prediction -= W[:, k][:, None] * (self.pi[k] @ self.X)[None]

        if self.X.ndim == 3:
            prediction = np.stack(
                [W[t] @ self.pi @ self.X[t] for t in range(self.dims['T'])])
            if k is not None:
                prediction -= np.stack(
                    [W[t, k] * (self.X[t] @ self.pi[:, k]) for t in range(self.dims['T'])]
                )
        return prediction

    def _compute_residual(self, k=None):
        """
        computes expected residual
        """
        prediction = self._compute_prediction(k)
        return self.Y - prediction

    def _update_pi_component(self, k):
        """
        update pi for a single component
        """
        
        # compute residual
        r_k = self._compute_residual(k)
        W = self.active * self.weight_means

        pi_k = np.zeros(self.dims['N'])
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(self.Y[tissue])
            tmp1 = -2 * (r_k[tissue, mask] @ self.X[:, mask].T) * (self.weight_means[tissue, k] * self.active[tissue, k])
            tmp2 = self.active[tissue, k] * np.einsum('ij,ij->i', self.X[:, mask], self.X[:, mask])
            pi_k += -0.5 / self.tissue_variance[tissue] * (tmp1 + tmp2)

        pi_k += np.log(self.pi_prior)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max() + 5)
        pi_k = pi_k / pi_k.sum()
        self.pi[k] = pi_k

    def _update_weight_component(self, k, ARD=True):
        """
        update weights for a component
        """
        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            if ARD:
                self.prior_weight_variance[tissue, k] = self.weight_vars[tissue, k] + self.weight_means[tissue, k]**2

            mask = ~np.isnan(self.Y[tissue])
            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

            precision = (self.pi[k] @ diag / self.tissue_variance[tissue]) + (1 / self.prior_weight_variance[tissue, k])
            variance = (1 / precision)

            mean = (variance / self.tissue_variance[tissue]) * (r_k[tissue, mask] @ (self.pi[k] @ self.X[:, mask]))
            self.weight_vars[tissue, k] = variance
            self.weight_means[tissue, k] = mean

    def _update_active_component(self, k, ARD=True):
        """
        update q(s_k)
        """
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_slab_weights[k] = (a - 1) / (a + b - 2)

        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(self.Y[tissue])
            mu2 = (self.weight_means[tissue, k] ** 2 + self.weight_vars[tissue, k]) * (self.pi[k] @ self.X[:, mask]**2)
            mu = (self.pi[k] @ self.X[:, mask]) * self.weight_means[tissue, k]
            trace = np.sum(mu2 - mu**2)

            off =  np.log(1 - self.prior_slab_weights[k]) \
                + np.sum(norm.logpdf(x=r_k[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue])))

            tmp1 = np.log(self.prior_slab_weights[k])
            tmp2 = np.sum(norm.logpdf(x=r_k[tissue, mask], loc=mu, scale=np.sqrt(self.tissue_variance[tissue])))
            tmp3 = - (0.5 / self.tissue_variance[tissue]) * trace
            tmp4 = - normal_kl(self.weight_means[tissue, k], self.weight_vars[tissue, k], 0.0, self.prior_weight_variance[tissue, k])
            on = tmp1 + tmp2 + tmp3 + tmp4
            self.active[tissue, k] = 1 / (1 + np.exp(-(on - off)))

    def _update_tissue_variance(self):
        residual = self._compute_residual()
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(residual[tissue])
            ERSS = np.sum(residual[tissue, mask] ** 2)
            # add trace term
            for component in range(self.dims['K']):
                mu2 = self.pi[component] @ (self.X[:, mask]**2) * self.active[tissue, component] \
                    * (self.weight_means[tissue, component]**2 + self.weight_vars[tissue,component])
                mu = self.pi[component] @ self.X[:, mask] \
                    * self.weight_means[tissue, component] * self.active[tissue, component]
                ERSS += 0.5 / self.tissue_variance[tissue] * np.sum(mu2 - mu**2)

            var = (ERSS / 2 + self.beta) / (self.alpha + mask.sum()/2 + 1)
            self.tissue_variance[tissue] = var

    def update_weights(self, components=None, ARD=True):
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_weight_component(k, ARD)

    def update_pi(self, components=None):
        """
        update pi
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_pi_component(k)

    def update_active(self, components=None, ARD=True):
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_active_component(k, ARD)

    def _fit(self, max_outer_iter=1000, max_inner_iter=1, bound=False, verbose=False, components=None, diffuse=1.0, update_weights=True, update_active=True, update_pi=True, update_prior_variance=False, update_prior_activity=False):
        """
        loop through updates until convergence
        """
        self.elbos.append(self.compute_elbo())
        for i in range(max_outer_iter):
            # update weights and activities
            for _ in range(max_inner_iter):
                if update_prior_variance:
                    pass

                if update_weights:
                    self.update_weights(components)

                if update_active:
                    self.update_active(components)

                self.elbos.append(self.compute_elbo())
                if np.abs(self.elbos[-1] - self.elbos[-2]) < self.tolerance:
                    break
            # update pi

            if update_pi:
                for _ in range(max_inner_iter):
                    self.update_pi()
                    self.elbos.append(self.compute_elbo())

                if np.abs(self.elbos[-1] - self.elbos[-2]) < self.tolerance:
                    break


            # after each inner loop record elbo
            self.elbos.append(self.compute_elbo())

            # check for convergence in ELBO
            if np.abs(self.elbos[-1] - self.elbos[-3]) < self.tolerance:
                if verbose:
                    print('ELBO converged at iter {}'.format(i))
                break

    def _forward_fit_step(self, l, max_inner_iter=1, max_outer_iter=1000,
                          diffuse=1.0, quantile=0.0,
                          bound=False, verbose=False, restarts=1, plots=False):
        """
        fit self as though there were only l components
        T initializations with unit weight at each tissue, pick best solution among them
        """
        K = self.dims['K']
        init_pi = self.pi.copy()
        init_active = self.active.copy()
        init_weights = self.weight_means.copy()

        restart_dict = {}
        elbos = []

        # initialize pi based on residuals
        residual = self._compute_residual()
        sq_err = np.max(residual**2, axis=0)
        pi = sq_err * (sq_err > np.quantile(sq_err, quantile))
        pi = residual.mean(0)
        pi = np.exp(pi - pi.max())
        pi = pi / pi.sum()

        self.pi[l-1] = pi

        #if plots:
        #    plt.scatter(np.arange(pi.size), self.pi[l-1])
        #    plt.show()

        # positive initialization
        for i in range(restarts):
            self.pi = init_pi
            self.active = init_active
            self.weights = init_weights

            # initialize activity to random
            active_t = init_active.copy()
            active_t[:, l-1] = np.random.random(self.dims['T'])

            # initialize weights something random
            weights_t = init_weights.copy()
            weights_t[:, l-1] = np.zeros_like(self.dims['T'])

            # initialize pi of the lth component, weigh snps with poor predictions heavier
            pi_t = init_pi.copy()
            pi_t[l-1] = pi

            self.active = active_t
            self.weights = weights_t
            self.pi = pi_t

            #self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l-1, l), diffuse=diffuse)

            # fit the whole model up to this component
            self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l-l, l), diffuse=diffuse)
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

    def compute_elbo(self):
        """
        copute evidence lower bound
        """

        expected_conditional = 0
        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        residual = self._compute_residual()
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(residual[tissue])
            expected_conditional += norm.logpdf(
                x=residual[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue])).sum()

            # add trace term
            for component in range(self.dims['K']):
                mu2 = self.pi[component] @ (self.X[:, mask]**2) * self.active[tissue, component] \
                    * (self.weight_means[tissue, component]**2 + self.weight_vars[tissue,component])
                mu = self.pi[component] @ self.X[:, mask] \
                    * self.weight_means[tissue, component] * self.active[tissue, component]
                expected_conditional -= 0.5 / self.tissue_variance[tissue] \
                    * np.sum(mu2 - mu**2)

        #TODO ADD lnp(prior_weight_variance) + lnp(prior_slab_weights)

        KL = 0
        # KL(q(W | S) || p(W)) = KL(q(W | S = 1) || p(W)) q(S = 1) + KL(p(W) || p(W)) (1 - q(S = 1))
        KL += np.sum(
            normal_kl(self.weight_means, self.weight_vars, 0, self.prior_weight_variance) * self.active
        )

        KL += np.sum(bernoulli_kl(self.active, self.prior_slab_weights[None]))
        KL += np.sum(
            [categorical_kl(self.pi[k], self.pi_prior) for k in range(self.dims['K'])]
        )

        return expected_conditional - KL
