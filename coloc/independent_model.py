import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl
import os, sys, pickle
from scipy.optimize import minimize_scalar
from .utils import np_cache_class, gamma_logpdf
from functools import lru_cache, cached_property
import time

class IndependentFactorSER:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip, get_expected_weights, check_convergence

    def __init__(self, X, Y, K, covariates=None, prior_activity=1.0, prior_variance=1.0, prior_pi=None, snp_ids=None, tissue_ids=None, sample_ids=None, tolerance=1e-5):
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
        self.prior_activity = np.ones(K) * prior_activity
        self.prior_pi = prior_pi  if (prior_pi is not None) else np.ones(N) / N

        # initialize latent vars
        self.active = np.ones((T, K))
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
        self.alpha0 = 1.0
        self.beta0 = 1e-10

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0

    def prior_variance(self):
        return 1 / self.prior_precision

    @lru_cache()
    def _get_mask(self, tissue):
        return ~np.isnan(self.Y[tissue])

    @lru_cache()
    def _get_diag(self, tissue):
        mask = self._get_mask(tissue)
        return np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])

    @np_cache_class(maxsize=2**5)
    def _compute_first_moment(self, pi, weight, active):
        return (pi * weight * active) @ self.X

    @lru_cache()
    def _compute_first_moment_hash(self, component, hash):
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        active = self.active[:, component][:, None]
        return self._compute_first_moment(pi, weight, active)

    @cached_property
    def credible_sets(self):
        self.get_credible_sets()[0]

    @cached_property
    def purity(self):
        self.get_credible_sets()[1]

    def compute_first_moment(self, component):
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        active = self.active[:, component][:, None]
        h = (pi @ weight.T).tobytes()
        return self._compute_first_moment_hash(component, h)

    @np_cache_class(maxsize=2**5)
    def _compute_second_moment(self, pi, weight, var, active):
        return (pi * (weight**2 + var) * active) @ self.X**2

    @lru_cache(maxsize=2**5)
    def _compute_second_moment_hash(self, component, hash):
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        var = self.weight_vars[:, component]
        active = self.active[:, component][:, None]
        return (pi * (weight**2 + var) * active) @ self.X**2

    def compute_second_moment(self, component):
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        var = self.weight_vars[:, component]
        active = self.active[:, component][:, None]

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
        update pi for a single component
        """
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual

        # 0 out nans
        r_k[np.isnan(self.Y)] = 0

        tmp1 = (-0.5 / self.tissue_variance[:, None]) * (
            - 2 * r_k @ (self.X.T) * self.weight_means[:, k]
            + self.weight_means[:, k] ** 2 * diag
        )
        tmp2 = -0.5 * (1 / self.tissue_variance[:, None]) * (self.weight_vars[:, k]) * diag
        tmp3 = -1 * normal_kl(self.weight_means[:, k], self.weight_vars[:, k], 0.0, self.prior_variance()[:, k][:, None])
        pi_k = (tmp1 + tmp2 + tmp3) * self.active[:, k][:, None]

        pi_k = pi_k.sum(0)
        pi_k += np.log(self.prior_pi)
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()

        self.pi[k] = pi_k

    def _update_weight_component(self, k, ARD=True, residual=None):
        """
        update weights for a component
        """
        if ARD:
            second_moment = (self.weight_vars[:, k] + self.weight_means[:, k]**2) @ self.pi[k] 
            alpha = self.alpha0 + 0.5
            beta = self.beta0 + second_moment / 2
            self.prior_precision[:, k] = np.clip((alpha - 1) / beta, 1e-10, 1e5)
        
        mask = np.isnan(self.Y)
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual
        r_k[mask] = 0
        
        precision = (diag / self.tissue_variance[:, None]) + (1 / self.prior_variance()[:, k])[:, None]
        variance = 1 / precision
        mean = (variance / self.tissue_variance[:, None]) * (r_k @ self.X.T)
        self.weight_vars[:, k] = variance
        self.weight_means[:, k] = mean

    def _update_active_component(self, k, ARD=True, precomputed_residual=None):
        """
        update q(s_k)
        """
        if ARD:
            a = self.active[:, k].sum() + 1
            b = 1 - self.active[:, k].sum() + self.dims['T']
            self.prior_activity[k] = (a - 1) / (a + b - 2)

        """
        mask = ~np.isnan(self.Y)
        diag = np.array([self._get_diag(k) for t in range(self.dims['T'])])
        r_k = self.compute_residual(k)
        r_k[mask] = 0
        off = np.sum(norm.logpdf(x=r_k, loc=0.0, scale=np.sqrt(self.tissue_variance)[:, None])) \
            + np.log(1 - self.prior_activity[k] + 1e-10)
        tmpa = self.weight_means[:, k] * self.X
        tmpa[mask] = 0
        tmp1 = (-0.5 * np.log(2 * np.pi * self.tissue_variance) * mask.sum(1)) - 0.5 / self.tissue_variance *
            ((r_k - tmpa) ** 2).sum(1)
        tmp2 = (-0.5 / self.tissue_variance) * (self.weight_vars[:, k]) * diag
        tmp3 = -1 * normal_kl(self.weight_means[:, k],
                              self.weight_vars[:, k],
                              0.0,
                              self.prior_variance()[tissue, k])
        on = np.inner(tmp1 + tmp2 + tmp3, self.pi[k]) + np.log(self.prior_activity[k] + 1e-10)
        """
        self.active[tissue, k] = np.clip(1 / (1 + np.exp(-(on - off))), 1e-5, 1-1e-5)

        for tissue in range(self.dims['T']):
            mask = ~np.isnan(self.Y[tissue])
            off = np.sum(norm.logpdf(x=r_k[tissue, mask], loc=0.0, scale=np.sqrt(self.tissue_variance[tissue]))) \
                + np.log(1 - self.prior_activity[k] + 1e-10)

            diag = np.einsum('ij, ij->i', self.X[:, mask], self.X[:, mask])
            tmp1 = (-0.5 * np.log(2 * np.pi * self.tissue_variance[tissue]) - 0.5 / self.tissue_variance[tissue] *
                (r_k[tissue, mask][None] - self.weight_means[tissue, k][:, None] * self.X[:, mask]) ** 2).sum(1)
            tmp2 = (-0.5 / self.tissue_variance[tissue]) * (self.weight_vars[tissue, k]) * diag
            tmp3 = -1 * normal_kl(self.weight_means[tissue, k],
                                  self.weight_vars[tissue, k],
                                  0.0,
                                  self.prior_variance()[tissue, k])
            on = np.inner(tmp1 + tmp2 + tmp3, self.pi[k]) + np.log(self.prior_activity[k] + 1e-10)

            self.active[tissue, k] = np.clip(1 / (1 + np.exp(-(on - off))), 1e-5, 1-1e-5)

    def _update_tissue_variance(self, residual=None):
        if residual is None:
            residual = self.compute_residual()
        ERSS = self._compute_ERSS(residual=residual)
        self.tissue_variance = ERSS / np.array([self._get_mask(t).sum() for t in range(self.dims['T'])])

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


    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_active=False, update_pi=True, update_variance=True, ARD_weights=False, ARD_active=False, update_covariate_weights=True):
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
                if update_weights: self._update_weight_component(l, ARD=ARD_weights, residual=residual)        
                if update_pi: self._update_pi_component(l, residual=residual)
                # if update_active: self._update_active_component(l, ARD=ARD_active)
                residual = residual - self.compute_first_moment(l)

            # update variance parameters
            if update_variance: self._update_tissue_variance(residual=residual)

            # monitor convergence with ELBO
            self.elbos.append(self.compute_elbo(residual=residual))
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
        # KL(q(W | S) || p(W)) = KL(q(W | S = 1) || p(W)) q(S = 1) + KL(p(W) || p(W)) (1 - q(S = 1))
        KL += np.sum(
            normal_kl(self.weight_means, self.weight_vars, 0, self.prior_variance()[..., None]) 
            * (self.active[..., None] * self.pi[None])
        )

        KL += np.sum(bernoulli_kl(self.active, self.prior_activity[None]))
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )
        KL += np.sum(gamma_logpdf(self.prior_precision, self.alpha0, self.beta0))
        #TODO ADD lnp(prior_weight_variance) + lnp(prior_slab_weights)

        return expected_conditional - KL

    def sort_components(self):
        """
        sort components by maximum component weight
        """
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

    def save_json(self, json_path, save_data=False):
        with gzip.GzipFile(json_path, 'w') as fout:
            fout.write(json.dumps(self.__dict__).encode('utf-8'))

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
