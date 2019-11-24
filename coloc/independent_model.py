import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar

class SpikeSlabSer:
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

    def _compute_prediction(self, k=None):
        """
        compute expected prediction
        E[XB]
        """
        W = self.weight_means * self.active
        if self.X.ndim == 2:
            # [T, K] @ ([M, N] @ [N, K]).T = [T, M]
            prediction = W @ (self.X @ self.pi).T
            if k is not None:
                prediction -= W[:, k][:, None] * (self.X @ self.pi[:, k])[None]

        if self.X.ndim == 3:
            prediction = np.stack(
                [W[t] @ (self.X[t] @ self.pi).T for t in range(self.dims['T'])])
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
            pi_k += -0.5 / self.tissue_variance[tissue] * (
                -2 * (r_k[tissue] @ self.X.T) * (self.weight_means[tissue] * self.active[tissue])[None] \
                + self.active * np.einsum('ij,ij->i', self.X, self.X)[:, None] * (self.active[tissue] * (self.weight_means[tissue]**2 + self.weight_vars[tissue]))[None]
            )
        pi_k += np.log(self.pi_prior)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max() + 5)
        pi_k = pi_k / pi_k.sum()
        self.pi[:, k] = pi_k

    def _update_weight_component(self, k):
        """
        update weights for a component
        """
        r_k = self._compute_residual(k)
        diag = np.einsum('ij, ij->i', self.X, self.X)

        precision = (self.pi[k] @ diag / self.tissue_variance) + (1 / self.prior_weight_variance[:, k])
        variance = (1 / precision)
        mean = (variance / self.tissue_variance) * (r_k @ self.X @ self.pi[k])
        self.weight_vars[:, k] = variance
        self.weight_means[:, k] = mean

    def _update_active_component(self, k):
        """
        update q(s_k)
        """
        r_k = self._compute_residual(k)
        for tissue in range(self.dims['T']):
            mu2 = (self.weight_means[tissue, k] ** 2 + self.weight_vars[tissue, k]) * (self.X**2 @ self.pi[k])
            mu = (self.X @ self.pi[k]) * self.weight_means[tissue, k]
            trace = np.sum(mu2 - mu**2)

            off = np.log(1 - self.prior_slab_weights[k])

            on = np.log(self.prior_slab_weights[k]) \
                + np.sum(norm.logpdf(x=r_k[tissue], loc=mu, scale=np.sqrt(self.tissue_variance[self.tissue_variance]))) \
                - (0.5 / self.tissue_variance[tissue]) * trace \
                - normal_kl(self.weight_means[tissue, k], self.weight_vars[tissue, k], 0.0, self.prior_weight_variance[tissue, k])
            self.active[tissue, k] = 1 / (1 + np.exp(-(on - off)))

    def update_weights(self):
        for k in range(self.dims['K']):
            self._update_weight_component(k)

    def update_pi(self, components=None):
        """
        update pi
        """
        for k in range(self.dims['K']):
            self._update_pi_component(k)

    def update_active(self):
        for k in range(self.dims['K']):
            self._update_active_component(k)

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
                x=residual[mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue])).sum()

            # add trace term
            for component in range(self.dims['K']):
                mu2 = (self.X**2) @ self.pi[component] * self.active[tissue, component] \
                    * (self.weight_means[tissue, component]**2 + self.weight_vars[tissue,component])
                mu = self.X @ self.pi[component] \
                    * self.weight_means[tissue, component] * self.active[tissue, component]
                expected_conditional -= 0.5 / self.tissue_variance[tissue] \
                    * np.sum(mu2 - mu**2)

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
