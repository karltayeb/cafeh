import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl, normal_entropy, gamma_kl
import os, sys, pickle
from scipy.special import digamma
from .utils import np_cache_class, gamma_logpdf
from functools import lru_cache
import time

class IndependentFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip, get_expected_weights, check_convergence

    def __init__(self, X, Y, K, covariates=None, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, sample_ids=None, tolerance=1e-5):
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

        self.prior_precision = np.ones((T, K))
        self.prior_pi = prior_pi  if (prior_pi is not None) else np.ones(N) / N

        # initialize latent vars
        self.weight_means = np.zeros((T, K, N))
        self.weight_vars = np.ones((T, K, N))
        self.tissue_variance = np.ones(T)
        self.pi = np.ones((K, N)) / N

        if self.covariates is not None:
            self.cov_weights = {tissue: np.zeros((covariates[tissue].shape[0])) for tissue in self.tissue_ids}
            self.sample_covariate_map = {tissue: np.isin(
                self.sample_ids, self.covariates[tissue].columns) for tissue in self.tissue_ids}
        else:
            self.cov_weights = None

        # hyper-parameters
        self.a = 1e-6
        self.b = 1e-6

        self.weight_precision_a = np.ones((T, K))
        self.weight_precision_b = np.ones((T, K))

        self.c = 1e-6
        self.d=1e-6

        self.tissue_precision_a = np.ones(T)
        self.tissue_precision_b = np.ones(T)

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0

    @property
    def expected_tissue_precision(self):
        """
        expected precision for tissue under variational approximation
        """
        return self.tissue_precision_a / self.tissue_precision_b

    @property
    def expected_weight_precision(self):
        """
        expected precision for weights under variational approximation
        """
        return self.weight_precision_a / self.weight_precision_b

    @property
    def credible_sets(self):
        """
        return credible sets
        """
        return self.get_credible_sets()[0]

    @property
    def purity(self):
        """
        return minimum absolute correlation of snps in each credible set
        """
        return self.get_credible_sets()[1]

    @lru_cache()
    def _get_mask(self, tissue):
        """
        nan mask to deal with missing values
        """
        return ~np.isnan(self.Y[tissue])

    @lru_cache()
    def _get_diag(self, tissue):
        """
        get diag(X^T X) for a given tissue
        differs for tissues because of missingness in Y
        """
        mask = self._get_mask(tissue)
        return np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

    @lru_cache()
    def _compute_first_moment_hash(self, component, hash):
        """
        compute E[Xzw] for a component
        """
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        return (pi * weight) @ self.X

    def compute_first_moment(self, component):
        """
        compute E[Xzw] for a component
        """
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        h = (pi @ weight.T).tobytes()
        return self._compute_first_moment_hash(component, h)

    @lru_cache(maxsize=2**5)
    def _compute_second_moment_hash(self, component, hash):
        """
        compute E[(Xzw)^2] for a component
        """
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        var = self.weight_vars[:, component]
        return (pi * (weight**2 + var)) @ self.X**2

    def compute_second_moment(self, component):
        """
        compute E[(Xzw)^2] for a component
        """
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        var = self.weight_vars[:, component]
        h = (pi @ (weight + var**2).T).tobytes()
        return self._compute_second_moment_hash(component, h)

    def _compute_covariate_prediction(self, compute=True):
        """
        predict from covariates
        compute is a boolean of whether to predict or return 0
            exists to clean up stuff in compute_prediction
        """
        prediction = np.zeros_like(self.Y)
        if (self.covariates is not None) and compute:
            for i, tissue in enumerate(self.tissue_ids):
                prediction[i, self.sample_covariate_map[tissue]] = \
                    self.cov_weights[tissue] @ self.covariates[tissue].values
        return prediction

    def compute_prediction(self, k=None, use_covariates=True):
        """
        compute expected prediction
        """
        prediction = self._compute_covariate_prediction(use_covariates)
        for l in range(self.dims['K']):
            prediction += self.compute_first_moment(l)
        if k is not None:
            prediction -= self.compute_first_moment(k)
        return prediction

    def compute_residual(self, k=None, use_covariates=True):
        """
        computes expected residual
        """
        prediction = self.compute_prediction(k, use_covariates)
        return self.Y - prediction

    def _compute_ERSS(self, residual=None):
        if residual is None:
            residual = self.compute_residual()
        ERSS = np.array([np.sum(residual[tissue, self._get_mask(tissue)]**2) for tissue in range(self.dims['T'])])
        for k in range(self.dims['K']):
            #mu = np.array([self._compute_first_moment(t, k) for t in range(self.dims['T'])])
            #mu2 = np.array([self._compute_second_moment(t, k) for t in range(self.dims['T'])])
            mu = self.compute_first_moment(k)
            mu2 = self.compute_second_moment(k)
            for t in range(self.dims['T']):
                mask = self._get_mask(t)
                ERSS[t] += mu2[t, mask].sum() - (mu[t, mask]**2).sum()
        return ERSS

    def _update_covariate_weights_tissue(self, residual, tissue):
        weights = np.zeros_like(self.cov_weights[tissue])
        Y = np.squeeze(residual[self.tissue_ids == tissue, self.sample_covariate_map[tissue]])
        X = self.covariates[tissue].values
        self.cov_weights[tissue] = np.linalg.pinv(X.T) @ Y

    def _update_pi_component(self, k, residual=None):
        """
        update pi for a component
        """
        mask = np.isnan(self.Y)
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])
        
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual
        r_k[mask] = 0


        # E[ln p(y | w, z, alpha , tau)]
        tmp1 = -2 * r_k @ self.X.T * self.weight_means[:, k] \
            + diag * (self.weight_means[:, k]**2 + self.weight_vars[:, k])
        tmp1 = -0.5 * self.expected_tissue_precision[:, [k]] * tmp1

        # E[ln p(w | alpha)]
        tmp2 = -0.5 * self.expected_weight_precision[:, k] * (
            self.weight_means[:, k]**2 + self.weight_vars[:, k]
        )

        # H(q(w))
        tmp3 = normal_entropy(self.weight_vars[:, k])

        import pdb; pdb.set_trace()
        pi_k = (tmp1 + tmp2 + tmp3)

        pi_k = pi_k.sum(0)
        pi_k += np.log(self.prior_pi)
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()

        self.pi[k] = pi_k

    def update_pi(self, components=None):
        """
        update pi
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_pi_component(k)


    def update_ARD_weights(self, k):
        """
        ARD update for weights
        """
        second_moment = (self.weight_vars[:, k] + self.weight_means[:, k]**2) @ self.pi[k]
        alpha = self.a + 0.5
        beta = self.b + second_moment / 2
        self.weight_precision_a[:, k] = alpha
        self.weight_precision_b[:, k] = beta

    def _update_weight_component(self, k, ARD=True, residual=None):
        """
        update weights for a component
        """
        mask = np.isnan(self.Y)
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])
        
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual
        r_k[mask] = 0

        precision = diag * self.expected_tissue_precision[:, None] \
            + self.expected_weight_precision[:, [k]]
        variance = 1 / precision  # [T, N]

        mean = (variance * self.expected_tissue_precision[:, None]) * (r_k @ self.X.T)
        self.weight_vars[:, k] = variance
        self.weight_means[:, k] = mean

    def update_weights(self, components=None, ARD=True):
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_weight_component(k, ARD)

    def update_tissue_variance(self, residual=None):
        if residual is None:
            residual = self.compute_residual()
        ERSS = self._compute_ERSS(residual=residual)
        self.tissue_variance = ERSS / np.array(
            [self._get_mask(t).sum() for t in range(self.dims['T'])])

    def update_covariate_weights(self):
        if self.covariates is not None:
            residual = self.compute_residual(use_covariates=False)
            for tissue in self.tissue_ids:
                self._update_covariate_weights_tissue(residual, tissue)

    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_pi=True, update_variance=True, ARD_weights=False, update_covariate_weights=True):
        """
        loop through updates until convergence
        """
        init_time = time.time()
        if components is None:
            components = np.arange(self.dims['K'])

        residual = self.compute_residual(use_covariates=True)
        for i in range(max_iter):
            # update covariate weights
            residual += self._compute_covariate_prediction(True)
            if (self.covariates is not None) and update_covariate_weights:
                self.update_covariate_weights()
            residual -= self._compute_covariate_prediction(True)

            # update component parameters
            for l in components:
                residual = residual + self.compute_first_moment(l)
                if update_weights: self._update_weight_component(
                    l, ARD=ARD_weights, residual=residual)
                if update_pi: self._update_pi_component(l, residual=residual)
                residual = residual - self.compute_first_moment(l)

            # update variance parameters
            if update_variance: self._update_tissue_variance(residual=residual)

            # monitor convergence with ELBO
            self.elbos.append(self.compute_elbo(residual=residual))
            if verbose:
                print("Iter {}: {}".format(i, self.elbos[-1]))

            cur_time = time.time()
            if self.check_convergence():
                if verbose:
                    print('ELBO converged with tolerance {} at iter: {}'.format(self.tolerance, i))
                break

        self.run_time += cur_time - init_time
        if verbose:
            print('cumulative run time: {}'.format(self.run_time))

    def compute_elbo(self, residual=None):
        """
        copute evidence lower bound
        """

        expected_conditional = 0
        KL = 0

        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        # compute expected conditional log likelihood E[ln p(Y | X, Z)]
        ERSS = self._compute_ERSS(residual=residual)
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            expected_conditional += -0.5 * mask.sum() * np.log(2 * np.pi * self.tissue_variance[tissue]) \
                -0.5 / self.tissue_variance[tissue] * ERSS[tissue]
        
        # <ln p (w | alpha) > - <ln q(w)> - <ln q(w)>
        Elna = digamma(self.weight_precision_a) - np.log(self.weight_precision_b)  # [T, K]
        precision = self.expected_weight_precision()  # [T, K]
        w2 = ((self.weight_means**2 + self.weight_vars) * self.pi[None]).sum(-1)  # [T, K]
        entropy = (normal_entropy(self.weight_vars) * self.pi[None]).sum(-1)  # [T, K]
        KL += np.sum(
            -0.5 * Elna  - 0.5 * precision * w2 - entropy
        )
        KL += Elna.sum()

        # Kl alpha
        # KL += gamma_kl(self.weight_precision_a, self.weight_precision_b, 1e-6, 1e-6).sum()

        # KL z
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )
        return expected_conditional - KL

    def get_ld(self, snps):
        return np.atleast_2d(np.corrcoef(self.X[snps.astype(int)]))

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
