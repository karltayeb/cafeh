import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar
import functools
from .utils import np_cache, np_cache_class, pack, unpack

class IndependentFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip, get_expected_weights

    def __init__(self, X, Y, K, covariates=None, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, sample_ids=None, tolerance=1e-3):
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
        self.covariates = covariates

        # set priors
        T, M = Y.shape
        N = X.shape[0]
        self.dims = {'N': N, 'M': M, 'T': T, 'K': K}

        self.tissue_ids = tissue_ids if (tissue_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)
        self.sample_ids = sample_ids if (sample_ids is not None) else np.arange(M)

        self.prior_variance = np.ones((T, K)) * prior_variance
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi or np.ones(N) / N

        # initialize latent vars
        self.active = np.ones((T, K)) * 0.99
        self.weight_means = np.zeros((T, K, N))
        self.weight_vars = np.ones((T, K, N))
        self.tissue_variance = np.ones(T)
        self.pi = np.ones((K, N)) / N

        if self.covariates is not None:
            self.cov_weights = {tissue: np.zeros((covariates[tissue].shape[0])) for tissue in self.tissue_ids}
        else:
            self.cov_weights = None

        # hyper-parameters
        self.alpha = 1e-10
        self.beta = 1e-10

        self.elbos = []
        self.tolerance = 1e-3

        self.moments = {'mu1': np.zeros((T, K, M)), 'mu2': np.ones((T, K, M))}

    def _compute_covariate_prediction(self, compute=True):
        """
        predict from covariates
        compute is a boolean of whether to predict or return 0
            exists to clean up stuff in _compute_prediction
        """
        prediction = np.zeros_like(self.Y)
        if (self.covariates is not None) and compute:
            for i, tissue in enumerate(self.tissue_ids):
                prediction[i, np.isin(self.sample_ids, self.covariates[tissue].columns)] = \
                    self.cov_weights[tissue] @ self.covariates[tissue].values
        return prediction

    def _compute_prediction_component(self, k):
        """
        compute prediction for a component
        """
        active = self.active[:, k]
        weights = self.weight_means[:, k]
        pi = self.pi[k]
        return (active[:, None] * weights * pi) @ self.X

    def _compute_prediction_tissue(self, tissue, k=None):
        """
        compute expected prediction
        """
        prediction = self._compute_covariate_prediction()[tissue]
        prediction += self.active[tissue] @ ((self.pi * self.weight_means[tissue]) @ self.X)
        if k is not None:
            prediction -= self.active[tissue, k] * (self.pi[k] * self.weight_means[tissue, k]) @ self.X
        return prediction

    def _compute_prediction_cache(self, k=None, use_covariates=True):
        """
        compute expected prediction
        """
        prediction = self._compute_covariate_prediction(use_covariates)
        for tissue in range(self.dims['T']):
            for component in range(self.dims['K']):
                prediction[tissue] += self.compute_first_moment(component, tissue)
            if k is not None:
                prediction[tissue] -= self.compute_first_moment(k, tissue)
        return prediction

    def compute_prediction(self, k=None, use_covariates=True):
        prediction = self._compute_covariate_prediction(use_covariates)
        for l in range(self.dims['K']):
            if l is not k: 
                prediction += self._compute_prediction_component(l)
        return prediction

    def compute_residual(self, k=None, use_covariates=True):
        """
        computes expected residual
        """
        prediction = self.compute_prediction(k, use_covariates)
        return self.Y - prediction

    @functools.lru_cache()
    def _get_mask(self, tissue):
        return ~np.isnan(self.Y[tissue])

    @functools.lru_cache()
    def _get_diag(self, tissue):
        mask = self._get_mask(tissue)
        return np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])


    @np_cache_class(maxsize=2**10)
    def _compute_first_moment(self, pi, weight, active):
        return (pi * weight * active) @ self.X

    def compute_first_moment(self, component, tissue):
        pi = self.pi[component]
        weight = self.weight_means[tissue, component]
        var = self.weight_vars[tissue, component]
        active = self.active[tissue, component]
        return self._compute_first_moment(pi, weight, active)

    @np_cache_class(maxsize=2**10)
    def _compute_second_moment(self, pi, weight, var, active):
        """
        mask = ~np.isnan(self.Y[tissue])
        pi = self.pi[component]
        weight = self.weight_means[tissue, component]
        var = self.weight_vars[tissue, component]
        active = self.active[tissue, component]
        X = self.X[:, mask]
        """
        # pi, weight, var, active, X = unpack(flat, sizes, shapes)
        return (pi * (weight**2 + var) * active) @ self.X**2

    def compute_second_moment(self, component, tissue):
        pi = self.pi[component]
        weight = self.weight_means[tissue, component]
        var = self.weight_vars[tissue, component]
        active = self.active[tissue, component]
        return self._compute_second_moment(pi, weight, var, active)

    def _compute_moments(self, tissue, component):
        """
        first and second moment of tissue, component prediction
        E[(x^T z w s)], E[(x^T z w s)^2]
        """
        mask = self._get_mask(tissue) # .astype(np.float64)

        pi = self.pi[component]
        weight = self.weight_means[tissue, component]
        var = self.weight_vars[tissue, component]
        active = self.active[tissue, component]

        mu = self._compute_first_moment(pi, weight, active)
        mu2 = self._compute_second_moment(pi, weight, active, var)
        return mu, mu2

    def _update_covariate_weights_tissue(self, residual, tissue):
        weights = np.zeros_like(self.cov_weights[tissue])
        Y = np.squeeze(residual[self.tissue_ids == tissue, np.isin(self.sample_ids, self.covariates[tissue].columns)])
        X = self.covariates[tissue].values
        self.cov_weights[tissue] = np.linalg.pinv(X.T) @ Y

    def _update_pi_component_new(self, k, precomputed_residual=None):
        """
        update pi for a single component
        """

        # compute residual
        r_k = precomputed_residual if (precomputed_residual is not None) else self.compute_residual(k)
        pi_k = np.zeros(self.dims['N'])
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            diag = self._get_diag(tissue)
            tmp1 = (1 / self.tissue_variance[tissue]) * (
                r_k[tissue, mask] @ (self.weight_means[tissue, k] * self.X[:, mask].T)
            )
            tmp2 = (-0.5 / self.tissue_variance[tissue]) * (
                (self.weight_means[tissue, k] ** 2 + self.weight_vars[tissue, k]) * diag
            )
            tmp3 = -1 * normal_kl(self.weight_means[tissue, k], self.weight_vars[tissue, k], 0.0, self.prior_variance[tissue, k])
            tmp1 = np.zeros(self.dims['N'])
            pi_k += (tmp1 + tmp2 + tmp3) * self.active[tissue, k]
        pi_k += np.log(self.prior_pi)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()
        self.pi[k] = pi_k

    def _update_pi_component(self, k, precomputed_residual=None):
        """
        update pi for a single component
        """

        # compute residual
        r_k = precomputed_residual if (precomputed_residual is not None) else self.compute_residual(k)
        pi_k = np.zeros(self.dims['N'])
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            diag = self._get_diag(tissue)
            #tmp1 = (-0.5 / self.tissue_variance[tissue] * (r_k[tissue, mask][None] - self.weight_means[tissue, k][:, None] * self.X[:, mask]) ** 2).sum(1)

            tmp1 = (-0.5 / self.tissue_variance[tissue]) * (
                -2 * r_k[tissue, mask] @ (self.weight_means[tissue, k] * self.X[:, mask].T)
            )
            tmp2 = -0.5 * (1 / self.tissue_variance[tissue]) * ((self.weight_means[tissue, k] ** 2) + self.weight_vars[tissue, k]) * diag
            tmp3 = -1 * normal_kl(self.weight_means[tissue, k], self.weight_vars[tissue, k], 0.0, self.prior_variance[tissue, k])
            pi_k += (tmp1 + tmp2 + tmp3) * self.active[tissue, k]
        pi_k += np.log(self.prior_pi)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()
        self.pi[k] = pi_k

    def _update_weight_component(self, k, ARD=True, precomputed_residual=None):
        """
        update weights for a component
        """
        r_k = precomputed_residual if (precomputed_residual is not None) else self.compute_residual(k)
        for tissue in range(self.dims['T']):
            if ARD:
                self.prior_variance[tissue, k] = np.inner(
                    (self.weight_vars[tissue, k] + self.weight_means[tissue, k]**2), self.pi[k]) * self.active[tissue, k]

            mask = self._get_mask(tissue)
            diag = self._get_diag(tissue)

            precision = (diag / self.tissue_variance[tissue]) + (1 / self.prior_variance[tissue, k])
            variance = 1 / precision
            mean = (variance / self.tissue_variance[tissue]) * (self.X[:, mask] @ r_k[tissue, mask].T)
            
            self.weight_vars[tissue, k] = variance
            self.weight_means[tissue, k] = mean

    def _update_active_component(self, k, ARD=True, precomputed_residual=None):
        """
        update q(s_k)
        """
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_activity[k] = (a - 1) / (a + b - 2)

        r_k = precomputed_residual if (precomputed_residual is not None) else self.compute_residual(k)
        
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            off = np.sum(norm.logpdf(x=r_k[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue]))) \
                + np.log(1 - self.prior_activity[k] + 1e-10)

            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])
            tmp1 = (-0.5 * np.log(2 * np.pi * self.tissue_variance[tissue]) - 0.5 / self.tissue_variance[tissue] *
                (r_k[tissue, mask][None] - self.weight_means[tissue, k][:, None] * self.X[:, mask]) ** 2).sum(1)
            tmp2 = (-0.5 / self.tissue_variance[tissue]) * (self.weight_vars[tissue, k]) * diag
            tmp3 = -1 * normal_kl(self.weight_means[tissue, k],
                                  self.weight_vars[tissue, k],
                                  0.0,
                                  self.prior_variance[tissue, k])
            on = np.inner(tmp1 + tmp2 + tmp3, self.pi[k]) + np.log(self.prior_activity[k] + 1e-10)

            self.active[tissue, k] = np.clip(1 / (1 + np.exp(-(on - off))), 1e-5, 1-1e-5)

    def _update_tissue_variance(self, precomputed_residual=None):
        residual = precomputed_residual if (precomputed_residual is not None) else self.compute_residual()
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            ERSS = np.sum(residual[tissue, mask] ** 2)
            # add trace term
            for component in range(self.dims['K']):
                mu, mu2 = self._compute_moments(tissue, component)
                #import pdb; pdb.set_trace()
                ERSS += np.sum(mu2[mask] - mu[mask]**2)

            # if you want to use inv gamma prior, but add it to likelihood
            # var = (ERSS / 2 + self.beta) / (self.alpha + mask.sum()/2 + 1)
            self.tissue_variance[tissue] = ERSS / mask.sum()

    def update_covariate_weights(self):
        if self.covariates is not None:
            residual = self.compute_residual(use_covariates=False)
            for tissue in self.tissue_ids:
                self._update_covariate_weights_tissue(residual, tissue)

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

    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_active=True, update_pi=True, update_variance=True, ARD_weights=False, ARD_active=False, update_covariate_weights=True):
        """
        loop through updates until convergence
        """
        if components is None:
            components = np.arange(self.dims['K'])

        self.elbos.append(self.compute_elbo())
        for i in range(max_iter):
            # update covariate weights
            if (self.covariates is not None) and update_covariate_weights:
                self.update_covariate_weights()

            # update component parameters
            r = self.compute_residual()
            for l in components:
                r_l = r + self._compute_prediction_component(l)
                if update_weights:
                    self._update_weight_component(l, ARD=ARD_weights, precomputed_residual=r_l)        
                if update_pi:
                    self._update_pi_component(l, precomputed_residual=r_l)
                if update_active:
                    self._update_active_component(l, ARD=ARD_active, precomputed_residual=r_l)

                r = r_l - self._compute_prediction_component(l)

            # update variance parameters
            if update_variance:
                self._update_tissue_variance(precomputed_residual=r)

            # monitor convergence with ELBO
            if i % 1 == 0:
                self.elbos.append(self.compute_elbo(precomputed_residual=r))
                if verbose:
                    print("Iter {}: {}".format(i, self.elbos[-1]))

                diff = self.elbos[-1] - self.elbos[-2]
                if (np.abs(diff) < self.tolerance):
                    if verbose:
                        print('ELBO converged with tolerance {} at iter: {}'.format(self.tolerance, i))
                    break

    def get_ld(self, snps):
        """
        compute the pairwise correlations for a set of snps
        snps is a np array of
        """
        ld = np.atleast_2d(np.corrcoef(self.X[snps]))
        return ld

    def forward_fit(self, max_iter=1000, verbose=False, update_weights=True, update_active=True, update_pi=True, ARD_weights=False, ARD_active=False):
        for k in range(self.dims['K']):
            self.fit(max_iter, verbose, np.arange(k), update_weights, update_active, update_pi, ARD_weights, ARD_active)

    def compute_elbo(self, precomputed_residual=None):
        """
        copute evidence lower bound
        """

        expected_conditional = 0
        KL = 0

        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        residual = precomputed_residual if (precomputed_residual is not None) else self.compute_residual()
        for tissue in range(self.dims['T']):
            mask = ~np.isnan(residual[tissue])
            expected_conditional += norm.logpdf(
                x=residual[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue])).sum()

            # add trace term
            for component in range(self.dims['K']):
                mu, mu2 = self._compute_moments(tissue, component)
                expected_conditional -= 0.5 / self.tissue_variance[tissue] \
                    * np.sum(mu2[mask] - mu[mask]**2)

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

    def save(self, output_dir, model_name):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

