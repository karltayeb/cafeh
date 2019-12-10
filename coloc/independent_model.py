import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class IndependentFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip

    def __init__(self, X, Y, K, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, tolerance=1e-6):
        """
        Y [T x M] expresion for tissue, individual
        X [N x M] genotype for snp, individual
            potentially [T x N x M] if tissue specific correction

        prior_weight_variance [T, K]
            prior variance for weight of (tissue, component) loading
        prior_activity [K]
            prior probability of sapling from slab in component k
        prior_pi: prior for multinomial,
            probability of sampling a snps as the active feature
        """

        # set data
        self.X = X
        self.Y = Y

        # set priors
        T, M = Y.shape
        N = X.shape[0]
        self.dims = {'N': N, 'M': M, 'T': T, 'K': K}

        self.prior_variance = np.ones((T, K)) * prior_variance
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi or np.ones(N) / N

        # initialize latent vars
        self.active = np.ones((T, K)) * 0.99
        self.weight_means = np.ones((T, K, N))
        self.weight_vars = np.ones((T, K, N))
        self.tissue_variance = np.ones(T)
        self.pi = np.ones((K, N)) / N


        self.tissue_ids = tissue_ids or np.arange(T)
        self.snp_ids = snp_ids or np.arange(N)

        # hyper-parameters
        self.alpha = 1e-10
        self.beta = 1e-10

        self.elbos = []
        self.tolerance = 1e-3

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
        computes expected residual
        """
        prediction = self._compute_prediction(k)
        return self.Y - prediction

    def _compute_moments(self, tissue, component):
        """
        first and second moment of tissue, component prediction
        E[(x^T z w s)], E[(x^T z w s)^2]
        """
        mask = ~np.isnan(self.Y[tissue])
        mu2 = (self.pi[component]
               * (self.weight_means[tissue, component]**2 + self.weight_vars[tissue, component])
               * self.active[tissue, component]) @ (self.X[:, mask]**2)
        mu = (self.pi[component]
              * self.weight_means[tissue, component]
              * self.active[tissue, component]) @ self.X[:, mask]
        return mu, mu2

    def _update_pi_component(self, k):
        """
        update pi for a single component
        """

        # compute residual
        r_k = self._compute_residual(k)
        pi_k = np.zeros(self.dims['N'])
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(self.Y[tissue])
            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

            tmp1 = norm.logpdf(
                r_k[tissue, mask][None],
                loc=(self.weight_means[tissue, k][:, None] * self.X),
                scale=np.sqrt(self.tissue_variance[tissue])).sum(1)
            tmp2 = -0.5 * (1 / self.tissue_variance[tissue]) * (self.weight_vars[tissue, k]) * diag
            tmp3 = -1 * normal_kl(self.weight_means[tissue, k], self.weight_vars[tissue, k], 0.0, self.prior_variance[tissue, k])
            pi_k += (tmp1 + tmp2 + tmp3) * self.active[tissue, k]
        pi_k += np.log(self.prior_pi)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()
        self.pi[k] = pi_k

    def _update_weight_component(self, k, ARD=True):
        """
        update weights for a component
        """
        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            if ARD:
                self.prior_variance[tissue, k] = np.inner(
                    (self.weight_vars[tissue, k] + self.weight_means[tissue, k]**2), self.pi[k]) * self.active[tissue, k]

            mask = ~np.isnan(self.Y[tissue])
            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

            precision = (diag / self.tissue_variance[tissue]) + (1 / self.prior_variance[tissue, k])
            variance = 1 / precision
            mean = (variance / self.tissue_variance[tissue]) * (self.X[:, mask] @ r_k[tissue, mask].T)
            
            self.weight_vars[tissue, k] = variance
            self.weight_means[tissue, k] = mean

    def _update_active_component(self, k, ARD=True):
        """
        update q(s_k)
        """
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_activity[k] = (a - 1) / (a + b - 2)

        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(self.Y[tissue])
            off = np.sum(norm.logpdf(x=r_k[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue]))) \
                + np.log(1 - self.prior_activity[k])

            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

            tmp1 = norm.logpdf(r_k[tissue, mask][None],
                               loc=(self.weight_means[tissue, k][:, None] * self.X),
                               scale=np.sqrt(self.tissue_variance[tissue])).sum(1)
            tmp2 = (-0.5 / self.tissue_variance[tissue]) * (self.weight_vars[tissue, k]) * diag

            tmp3 = -1 * normal_kl(self.weight_means[tissue, k],
                                  self.weight_vars[tissue, k],
                                  0.0,
                                  self.prior_variance[tissue, k])
            on = np.inner(tmp1 + tmp2 + tmp3, self.pi[k]) + np.log(self.prior_activity[k])

            self.active[tissue, k] = np.clip(1 / (1 + np.exp(-(on - off))), 1e-5, 1-1e-5)

    def _update_tissue_variance(self):
        residual = self._compute_residual()
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(residual[tissue])
            ERSS = np.sum(residual[tissue, mask] ** 2)
            # add trace term
            for component in range(self.dims['K']):
                mu, mu2 = self._compute_moments(tissue, component)
                ERSS += np.sum(mu2 - mu**2)

            # if you want to use inv gamma prior, but add it to likelihood
            # var = (ERSS / 2 + self.beta) / (self.alpha + mask.sum()/2 + 1)
            self.tissue_variance[tissue] = ERSS / mask.sum()

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


    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_active=True, update_pi=True, update_variance=True, ARD_weights=False, ARD_active=False):
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
                if update_variance:
                    self._update_tissue_variance()

            self.elbos.append(self.compute_elbo())

            diff = self.elbos[-1] - self.elbos[-2]
            if (np.abs(diff) < self.tolerance):
                if verbose:
                    print('ELBO converged with tolerance {} at iter: {}'.format(self.tolerance, i))
                break

    def forward_fit(self, max_iter=1000, verbose=False, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        for k in range(self.dims['K']):
            self.fit(max_iter, verbose, np.arange(k), update_weights, update_active, update_pi, ARD_weights, ARD_active)

    def compute_elbo(self):
        """
        copute evidence lower bound
        """

        expected_conditional = 0
        KL = 0

        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        residual = self._compute_residual()
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(residual[tissue])
            expected_conditional += norm.logpdf(
                x=residual[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue])).sum()

            # add trace term
            for component in range(self.dims['K']):
                mu, mu2 = self._compute_moments(tissue, component)
                expected_conditional -= 0.5 / self.tissue_variance[tissue] \
                    * np.sum(mu2 - mu**2)

        # KL(q(W | S) || p(W)) = KL(q(W | S = 1) || p(W)) q(S = 1) + KL(p(W) || p(W)) (1 - q(S = 1))
        KL += np.sum(
            normal_kl(self.weight_means, self.weight_vars, 0, self.prior_variance[..., None]) 
            * (self.active[..., None] * self.pi[None])
        )

        KL += np.sum(bernoulli_kl(self.active, self.prior_activity[None]))
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )

        #TODO ADD lnp(prior_weight_variance) + lnp(prior_slab_weights)

        return expected_conditional - KL
