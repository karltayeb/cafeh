import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl, normal_entropy, gamma_kl
import os, sys, pickle
from scipy.special import digamma
from .utils import np_cache_class, gamma_logpdf
from functools import lru_cache
import pandas as pd
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
        self.covariates = pd.concat(covariates, sort=True).fillna(0)

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
        self.pi = np.ones((K, N)) / N

        if self.covariates is not None:
            self.cov_weights = {
                t: np.zeros(self.covariates.loc[t].shape[0])
                for t in self.tissue_ids
            }
        else:
            self.cov_weights = None

        # hyper-parameters
        self.a = 1e-10
        self.b = 1e-10

        self.weight_precision_a = np.ones((T, K))
        self.weight_precision_b = np.ones((T, K))

        self.c = 1e-10
        self.d=1e-10

        self.tissue_precision_a = np.ones(T)
        self.tissue_precision_b = np.ones(T)

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0


        masks = {t: ~np.isnan(self.Y[t]) for t in range(T)}
        diags = {t: np.einsum(
            'ij, ij->i', self.X[:, masks[t]], self.X[:, masks[t]]) for t in range(T)}

        if covariates is not None:
            cov_pinv = {t: np.linalg.pinv(self.covariates.loc[t].values.T) for t in self.tissue_ids}
        else:
            cov = {}

        self.precompute = {
            'Hw': {},
            'Ew2': {},
            'first_moments': {},
            'diags': diags,
            'masks': masks,
            'cov_pinv': cov_pinv,
            'covariate_prediction': {}
        }

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
    def expected_effects(self):
        """
        computed expected effect size E[zw] [T, N]
        """
        return np.einsum('ijk,jk->ik', self.weight_means, self.pi)

    def compute_Hw(self, component):
        """
        compute entropy of q(w)
        """
        if component not in self.precompute['Hw']:
            v1 = self.weight_vars[:, component]
            self.precompute['Hw'][component] = normal_entropy(v1)
        return self.precompute['Hw'][component]

    def compute_Ew2(self, component):
        """
        compute second moment of q(w)
        """
        if component not in self.precompute['Ew2']:
            m1 = self.weight_means[:, component]
            v1 = self.weight_vars[:, component]
            self.precompute['Ew2'][component] = (m1**2 + v1)
        return self.precompute['Ew2'][component]

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

    def _get_mask(self, tissue):
        """
        nan mask to deal with missing values
        """
        return self.precompute['masks'][tissue]

    def _get_diag(self, tissue):
        """
        get diag(X^T X) for a given tissue
        differs for tissues because of missingness in Y
        """
        return self.precompute['diags'][tissue]

    def compute_first_moment(self, component):
        """
        compute E[Xzw] for a component
        """

        # if its not computed, compute now
        if component not in self.precompute['first_moments']:
            pi = self.pi[component]
            weight = self.weight_means[:, component]
            moment = (pi * weight) @ self.X
            self.precompute['first_moments'][component] = moment

        return self.precompute['first_moments'][component]

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

        prediction = []
        if (self.covariates is not None) and compute:
            for i, tissue in enumerate(self.tissue_ids):
                prediction.append(self.cov_weights[tissue] @ self.covariates.loc[tissue].values)
            prediction = np.array(prediction)
        else:
            prediction = np.zeros_like(self.Y)
        return prediction

    def compute_prediction(self, k=None, use_covariates=True):
        """
        compute expected prediction
        """
        prediction = self._compute_covariate_prediction(use_covariates)

        prediction = np.sum([
            self.compute_first_moment(l) for l in range(self.dims['K']) if l != k
        ], axis=0)
        return prediction

    def compute_residual(self, k=None, use_covariates=True):
        """
        computes expected residual
        """
        prediction = self.compute_prediction(k, use_covariates)
        return self.Y - prediction

    def _compute_ERSS(self, k=None):
        """
        compute ERSS using XY and XX
        """
        ERSS = np.zeros(self.dims['T'])

        prediction = self.compute_prediction(k, use_covariates=False)
        first_moments = np.array([
            self.compute_first_moment(l)
            for l in range(self.dims['K']) if l != k
        ])
        for t in range(self.dims['T']):
            mask = self._get_mask(t)
            diag = self._get_diag(t)

            pt1 = np.sum((self.weight_means[t] ** 2 + self.weight_vars[t]) * self.pi * diag)
            if k is not None:
                pt1 -= np.sum(self.compute_Ew2(k)[t] * self.pi[k] * diag)

            pt2 = np.inner(prediction[t, mask], prediction[t, mask])
            pt3 = np.einsum('ij,ij->i', first_moments[:, t, mask], first_moments[:, t, mask]).sum()
            ERSS[t] = np.inner(self.Y[t, mask], self.Y[t, mask])
            ERSS[t] += -2 * np.inner(self.Y[t, mask], prediction[t, mask])
            ERSS[t] += pt1 + np.sum(pt2) - pt3
        return ERSS

    def _update_covariate_weights_tissue(self, residual, tissue):
        """
        update covariates
        nans are masked with 0s-- same as filtering down to relevant
        samples
        """
        Y = np.squeeze(residual[self.tissue_ids == tissue])
        Y[np.isnan(Y)] = 0
        X = self.precompute['cov_pinv'][tissue]
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

        E_ln_alpha = digamma(self.weight_precision_a[:, k]) \
            - np.log(self.weight_precision_b[:, k])
        E_alpha = self.expected_weight_precision[:, k]
        E_w2 = self.compute_Ew2(k)

        # E[ln p(y | w, z, alpha , tau)]
        tmp1 = -2 * r_k @ self.X.T * self.weight_means[:, k] \
            + diag * E_w2
        tmp1 = -0.5 * self.expected_tissue_precision[:, None] * tmp1


        # E[ln p(w | alpha)] + H(q(w))
        lik = (
            -0.5 * np.log(2 * np.pi)
            + 0.5 * E_ln_alpha[:, None]
            - 0.5 * E_alpha[:, None] * E_w2)  # [T, N]
        entropy = self.compute_Hw(k)
        pi_k = (tmp1 + lik + entropy)

        pi_k = pi_k.sum(0)
        pi_k += np.log(self.prior_pi)
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()

        # update parameters
        self.pi[k] = pi_k

        # pop precomputes
        self.precompute['first_moments'].pop(k, None)
        self.precompute['Hw'].pop(k, None)
        self.precompute['Ew2'].pop(k, None)

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
            + self.expected_weight_precision[:, k][:, None]
        variance = 1 / precision  # [T, N]
        mean = (variance * self.expected_tissue_precision[:, None]) * (r_k @ self.X.T)

        # update params
        self.weight_vars[:, k] = variance
        self.weight_means[:, k] = mean

        # pop precomputes
        self.precompute['first_moments'].pop(k, None)
        self.precompute['Hw'].pop(k, None)
        self.precompute['Ew2'].pop(k, None)

    def update_weights(self, components=None, ARD=True):
        """
        update weights for all components
        """
        if components is None:
            components = np.arange(self.dims['K'])

        for k in components:
            self._update_weight_component(k, ARD)

    def update_tissue_variance(self, residual=None):
        """
        update tau, controls tissue specific variance
        """
        if residual is None:
            residual = self.compute_residual()
        ERSS = self._compute_ERSS()

        n_samples = np.array([
            self._get_mask(t).sum() for t in range(self.dims['T'])
        ])
        self.tissue_precision_a = self.c + n_samples / 2
        self.tissue_precision_b = self.d + ERSS / 2

    def update_covariate_weights(self):
        """
        update covariates
        """
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

        for i in range(max_iter):
            # update covariate weights
            if (self.covariates is not None) and update_covariate_weights:
                self.update_covariate_weights()

            # update component parameters
            for l in components:
                if ARD_weights:
                    self.update_ARD_weights(l)
                if update_weights:
                    self._update_weight_component(l)
                if update_pi:
                    self._update_pi_component(l)

            # update variance parameters
            if update_variance:
                self.update_tissue_variance()

            # monitor convergence with ELBO
            self.elbos.append(self.compute_elbo())
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

        E_ln_alpha = digamma(self.weight_precision_a) - np.log(self.weight_precision_b)
        E_alpha = self.expected_weight_precision

        E_ln_tau = digamma(self.tissue_precision_a) - np.log(self.tissue_precision_b)
        E_tau = self.expected_tissue_precision

        ERSS = self._compute_ERSS()
        for tissue in range(self.dims['T']):
            mask = self._get_mask(tissue)
            expected_conditional += \
                - 0.5 * mask.sum() * np.log(2 * np.pi) \
                + 0.5 * mask.sum() * E_ln_tau[tissue] \
                - 0.5 * E_tau[tissue] * ERSS[tissue]

        Ew2 = np.array([self.compute_Ew2(k) for k in range(self.dims['K'])])
        E_w2 = np.einsum('jik,jk->ij', Ew2, self.pi)

        Hw = np.array([self.compute_Hw(k) for k in range(self.dims['K'])])
        entropy = np.einsum('jik,jk->ij', Hw, self.pi)

        lik = (
            - 0.5 * np.log(2 * np.pi)
            + 0.5 * E_ln_alpha
            - 0.5 * E_alpha * E_w2
        )
        KL -= lik.sum() + entropy.sum()
        KL += gamma_kl(self.weight_precision_a, self.weight_precision_b, self.a, self.b).sum()
        KL += gamma_kl(self.tissue_precision_a, self.tissue_precision_b, self.c, self.d).sum()
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )
        return expected_conditional - KL

    def get_ld(self, snps):
        """
        ld matrix for subset of snps
        snps: integer index into snp_ids
        """
        return np.atleast_2d(np.corrcoef(self.X[snps.astype(int)]))

    def save(self, output_dir, model_name, save_data=False):
        """
        save the model
        """
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
