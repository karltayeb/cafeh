import numpy as np
from scipy.stats import norm
from .kls import unit_normal_kl, normal_kl, categorical_kl, bernoulli_kl, normal_entropy, gamma_kl
import os, sys, pickle
from scipy.special import digamma
from .utils import np_cache_class, gamma_logpdf, centered_moment2natural, natural2centered_moment
from functools import lru_cache
import time

class CAFEHGSimple:
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_pip, get_study_pip, get_expected_weights, check_convergence

    def __init__(self, X, Y, K, covariates=None, prior_variance=1.0, prior_pi=None, snp_ids=None, study_ids=None, sample_ids=None, tolerance=1e-5):
        """
        Y [T x M] expresion for study, individual
        X [N x M] genotype for snp, individual
            potentially [T x N x M] if study specific correction
        prior_weight_variance [T, K]
            prior variance for weight of (study, component) loading
        prior_activity [K]
            prior probability of sapling from slab in component k
        prior_pi: prior for multinomial,
            probability of sampling a snps as the active feature
        """

        # set data
        self.X = X
        self.Y = Y

        if covariates is not None:
            self.covariates = covariates.fillna(0)
        else:
            self.covariates = None

        # set priors
        T, M = Y.shape
        N = X.shape[0]
        self.dims = {'N': N, 'M': M, 'T': T, 'K': K}

        self.study_ids = study_ids if (study_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)
        self.sample_ids = sample_ids if (sample_ids is not None) else np.arange(M)

        self.prior_pi = prior_pi  if (prior_pi is not None) else np.ones(N) / N
        self.prior_activity = np.ones(K) * 0.5

        # initialize latent vars
        self.weight_means = np.zeros((T, K))
        self.weight_vars = np.ones((T, K))
        self.pi = np.ones((K, N)) / N
        self.active = np.ones((T, K))

        if self.covariates is not None:
            self.cov_weights = {
                t: np.zeros(self.covariates.loc[t].shape[0])
                for t in self.study_ids
            }
        else:
            self.cov_weights = None

        # hyper-parameters
        self.a = 1e-10
        self.b = 1e-10

        self.weight_precision_a = np.ones((T, K))
        self.weight_precision_b = np.ones((T, K))

        self.c = 1e-10
        self.d = 1e-10

        self.study_precision_a = np.ones(T)
        self.study_precision_b = np.nanvar(Y, 1)

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0
        self.step_size = 0.5

        masks = {t: ~np.isnan(self.Y[t]) for t in range(T)}
        self.diags = np.array([
            np.einsum('ij, ij->i',
                self.X[:, masks[t]], self.X[:, masks[t]]) for t in range(T)])

        if covariates is not None:
            cov_pinv = {t: np.linalg.pinv(self.covariates.loc[t].values.T) for t in self.study_ids}
        else:
            cov_pinv = {}

        self.precompute = {
            'Hw': {},
            'Ew2': {},
            'first_moments': {},
            'masks': masks,
            'cov_pinv': cov_pinv,
            'covariate_prediction': {}
        }

    @property
    def expected_study_precision(self):
        """
        expected precision for study under variational approximation
        """
        return self.study_precision_a / self.study_precision_b

    @property
    def expected_weight_precision(self):
        """
        expected precision for weights under variational approximation
        """
        return self.weight_precision_a / self.weight_precision_b

    @property
    def expected_log_odds(self):
        """
        computed expected effect size E[zw] [T, N]
        """
        return np.log(self.prior_activity + 1e-10) - np.log(1 - self.prior_activity + 1e-10)

    @property
    def expected_effects(self):
        """
        computed expected effect size E[zw] [T, N]
        """
        return np.einsum('ijk,jk->ik',
            self.weight_means * self.active[:, :, None], self.pi)

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

    def _get_mask(self, study):
        """
        nan mask to deal with missing values
        """
        return self.precompute['masks'][study]

    def _get_diag(self, study):
        """
        get diag(X^T X) for a given study
        differs for studys because of missingness in Y
        """
        return self.diags[study]

    def compute_first_moment_randomized(self, component, Q=10):
        """
        compute E[Xzw] for a component
        """
        # if its not computed, compute now
        if component not in self.precompute['first_moments']:
            pi = self.pi[component]
            weight = self.weight_means[:, component][:, None]
            active = self.active[:, component][:, None]
            sample = np.random.choice(a=pi.size, size=Q, p=pi)
            moment = self.X[sample].mean(0) * (weight * active)
            self.precompute['first_moments'][component] = moment
        return self.precompute['first_moments'][component]

    def compute_first_moment(self, component):
        """
        compute E[Xzw] for a component
        """

        # if its not computed, compute now
        if component not in self.precompute['first_moments']:
            pi = self.pi[component]
            weight = self.weight_means[:, component][:, None]
            active = self.active[:, component][:, None]
            moment = (pi @ self.X) * (weight * active)
            self.precompute['first_moments'][component] = moment
        return self.precompute['first_moments'][component]

    def compute_Hw(self, component):
        """
        compute entropy of q(w|z, s=1)
        """
        if component not in self.precompute['Hw']:
            v1 = self.weight_vars[:, component]
            self.precompute['Hw'][component] = normal_entropy(v1)
        return self.precompute['Hw'][component]

    def compute_Ew2(self, component):
        """
        compute second moment of q(w |z, s=1)
        """
        if component not in self.precompute['Ew2']:
            m1 = self.weight_means[:, component]
            v1 = self.weight_vars[:, component]
            self.precompute['Ew2'][component] = (m1**2 + v1)
        return self.precompute['Ew2'][component]

    def compute_covariate_prediction(self, compute=True):
        """
        predict from covariates
        compute is a boolean of whether to predict or return 0
            exists to clean up stuff in compute_prediction
        """

        prediction = []
        if (self.covariates is not None) and compute:
            for i, study in enumerate(self.study_ids):
                prediction.append(self.cov_weights[study] @ self.covariates.loc[study].values)
            prediction = np.array(prediction)
        else:
            prediction = np.zeros_like(self.Y)
        return prediction

    def compute_prediction(self, k=None, use_covariates=True):
        """
        compute expected prediction
        """
        prediction = self.compute_covariate_prediction(use_covariates)
        prediction += np.sum([
            self.compute_first_moment(l) for l in range(self.dims['K']) if l != k
        ], axis=0)
        return prediction

    def compute_residual(self, k=None, use_covariates=True):
        """
        computes expected residual
        """
        prediction = self.compute_prediction(k, use_covariates)
        return self.Y - prediction

    def _compute_ERSS(self):
        """
        compute ERSS using XY and XX
        """
        covariate_residual = self.Y - self.compute_covariate_prediction()

        ERSS = np.zeros(self.dims['T'])
        for t in range(self.dims['T']):
            mask = self._get_mask(t)
            diag = self._get_diag(t)

            active = self.active[t]  # Kx1
            Ew2 = (self.weight_means[t]**2 + self.weight_vars[t])  #KxN
            m2pid = active * Ew2 * (self.pi @ diag)
            mpX = (self.weight_means[t] * active)[:, None] * (self.pi @ self.X)

            y = covariate_residual[t, mask]
            ERSS[t] = np.inner(y, y)
            ERSS[t] += -2 * np.inner(y, mpX.sum(0))
            ERSS[t] += m2pid.sum() + np.sum(mpX.sum(0)**2) - np.sum(mpX**2)
        return ERSS

    def _update_covariate_weights_study(self, residual, study):
        """
        update covariates
        nans are masked with 0s-- same as filtering down to relevant
        samples
        """
        Y = np.squeeze(residual[self.study_ids == study])
        Y[np.isnan(Y)] = 0
        pinvX = self.precompute['cov_pinv'][study]
        self.cov_weights[study] = pinvX @ Y

    def _update_pi_component(self, k, residual=None):
        """
        update pi for a component
        """
        mask = np.isnan(self.Y)
        diag = self.diags
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual
        r_k[mask] = 0

        E_ln_alpha = digamma(self.weight_precision_a[:, k]) \
            - np.log(self.weight_precision_b[:, k])
        E_alpha = self.expected_weight_precision[:, k][:, None]
        E_tau = self.expected_study_precision[:, None]
        Ew2 = self.compute_Ew2(k)[:, None]
        active = self.active[:, k][:, None]

        # E[ln p(y | w, z, alpha , tau)]
        tmp1 = -2 * r_k @ self.X.T * self.weight_means[:, k][:, None] \
            + diag * Ew2
        tmp1 = -0.5 * E_tau * tmp1 * active

        pi_k = tmp1.sum(0)
        pi_k += np.log(self.prior_pi)
        pi_k = np.exp(pi_k - pi_k.max())
        pi_k = pi_k / pi_k.sum()

        # stochastic variational update
        if self.step_size < 1:
            natural_old = np.log(self.pi[k]) - np.log(self.pi[k, -1] + 1e-10)
            nautral_new = np.log(pi_k) - np.log(pi_k[-1] + 1e-10)
            natural_updated = (1 - self.step_size) * natural_old \
                + self.step_size * nautral_new
            pi_k = np.exp(natural_updated - natural_updated.max())
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
        active = self.active[:, k]
        E_alpha = self.expected_weight_precision[:, k]
        Ew2 = self.compute_Ew2(k)
        second_moment = Ew2 #s@ self.pi[k]
        second_moment = second_moment * active \
            + (1 / E_alpha) * (1 - active)
        alpha = self.a + 0.5
        beta = self.b + second_moment / 2

        if self.step_size < 1:
            alpha = (1 - self.step_size) * self.weight_precision_a[:, k] + \
                self.step_size * alpha
            beta = (1 - self.step_size) * self.weight_precision_b[:, k] + \
                self.step_size * beta
                
        self.weight_precision_a[:, k] = alpha
        self.weight_precision_b[:, k] = beta

    def _update_weight_component(self, k, residual=None):
        """
        update weights for a component
        """
        mask = np.isnan(self.Y)
        diag = self.diags
        d = diag @ self.pi[k]

        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual
        r_k[mask] = 0

        precision = d * self.expected_study_precision \
            + self.expected_weight_precision[:, k]
        variance = 1 / precision  # [T, N]
        mean = (variance * self.expected_study_precision) \
            * (r_k @ (self.X.T @ self.pi[k]))

        # stochastic optimization
        if self.step_size < 1:
            eta1_old, eta2_old = centered_moment2natural(self.weight_means[:, k], self.weight_vars[:, k])
            eta1_new, eta2_new = centered_moment2natural(mean, variance)

            eta1_updated = (1 - self.step_size) * eta1_old  +  self.step_size * eta1_new
            eta2_updated = (1 - self.step_size) * eta2_old  +  self.step_size * eta2_new
            mean, variance = natural2centered_moment(eta1_updated, eta2_updated)

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

    def _update_active_component(self, k):
        """
        update active
        """
        diag = self.diags # T x N
        d = diag @ self.pi[k]

        r_k = self.compute_residual(k)
        r_k[np.isnan(self.Y)] = 0
        p_k = self.compute_first_moment(k) / self.active[:, k][:, None]

        tmp1 = -2 * np.einsum('ij,ij->i', r_k, p_k) \
            + (self.compute_Ew2(k) * d)
        tmp1 = -0.5 * self.expected_study_precision * tmp1
        tmp2 = -0.5 * self.expected_weight_precision[:, k] \
            * self.compute_Ew2(k) \
            + normal_entropy(self.weight_vars[:, k])

        a = tmp1 + tmp2
        b = -0.5 + normal_entropy(1 / self.expected_weight_precision[:, k])
        active_k = 1 / (1 + np.exp(b - a - self.expected_log_odds[k]))
        # stochastic variational update
        if self.step_size < 1:
            natural_old = np.log(1 - self.active[:, k] + 1e-10) - np.log(self.active[:, k] + 1e-10)
            natural_new = b - a - self.expected_log_odds[k]

            natural_updated = (1 - self.step_size) * natural_old \
                + self.step_size * natural_new

            active_k = 1 / (np.exp(natural_updated) + 1)

        # update params
        self.active[:, k] = active_k

        # pop precomputes
        self.precompute['first_moments'].pop(k, None)
        self.precompute['Hw'].pop(k, None)
        self.precompute['Ew2'].pop(k, None)

    def update_study_variance(self, residual=None):
        """
        update tau, controls study specific variance
        """
        if residual is None:
            residual = self.compute_residual()
        ERSS = self._compute_ERSS()

        n_samples = np.array([
            self._get_mask(t).sum() for t in range(self.dims['T'])
        ])
        self.study_precision_a = self.c + n_samples / 2
        self.study_precision_b = self.d + ERSS / 2

    def update_covariate_weights(self):
        """
        update covariates
        """
        if self.covariates is not None:
            residual = self.compute_residual(use_covariates=False)
            for study in self.study_ids:
                self._update_covariate_weights_study(residual, study)

    def fit(self, max_iter=1000, verbose=False, components=None, **kwargs):
        """
        loop through updates until convergence
        """
        init_time = time.time()
        if components is None:
            components = np.arange(self.dims['K'])

        for i in range(max_iter):
            # update covariate weights
            if (self.covariates is not None) and kwargs.get('update_covariate_weights', True):
                self.update_covariate_weights()

            # update component parameters
            for l in components:
                if kwargs.get('ARD_weights', False):
                    self.update_ARD_weights(l)
                if kwargs.get('update_weights', True):
                    self._update_weight_component(l)
                if kwargs.get('update_pi', True):
                    self._update_pi_component(l)
                if kwargs.get('update_active', False):
                    self._update_active_component(l)

            # update variance parameters
            if kwargs.get('update_variance', False):
                self.update_study_variance()

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

        E_ln_tau = digamma(self.study_precision_a) - np.log(self.study_precision_b)
        E_tau = self.expected_study_precision

        ERSS = self._compute_ERSS()
        for study in range(self.dims['T']):
            mask = self._get_mask(study)
            expected_conditional += \
                - 0.5 * mask.sum() * np.log(2 * np.pi) \
                + 0.5 * mask.sum() * E_ln_tau[study] \
                - 0.5 * E_tau[study] * ERSS[study]

        Ew2 = np.array([self.compute_Ew2(k) for k in range(self.dims['K'])]).T
        Ew2 = Ew2 * self.active + (1 - self.active) / E_alpha

        Hw = np.array([self.compute_Hw(k) for k in range(self.dims['K'])]).T
        entropy = Hw * self.active + \
            normal_entropy(1 / E_alpha) * (1 - self.active)
        lik = (
            - 0.5 * np.log(2 * np.pi)
            + 0.5 * E_ln_alpha
            - 0.5 * E_alpha * Ew2
        )

        KL -= lik.sum() + entropy.sum()
        KL += gamma_kl(self.weight_precision_a, self.weight_precision_b, self.a, self.b).sum()
        KL += gamma_kl(self.study_precision_a, self.study_precision_b, self.c, self.d).sum()
        KL += np.sum(
            [categorical_kl(self.pi[k], self.prior_pi) for k in range(self.dims['K'])]
        )

        KL += np.sum([
            bernoulli_kl(self.active[:, k], self.prior_activity[k])
            for k in range(self.dims['K'])
        ])
        return expected_conditional - KL

    def get_ld(self, snps):
        """
        ld matrix for subset of snps
        snps: integer index into snp_ids
        """
        return np.atleast_2d(np.corrcoef(self.X[snps.astype(int)]))

    def _compress_model(self):
        """
        nothing to do
        """
        pass

    def _decompress_model(self):
        """
        nothing to do
        """
        pass

    def save(self, save_path, save_data=False):
        """
        save the model
        """
        # make save directory
        output_dir = '/'.join(save_path.split('/')[:-1])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # empty out model data
        for key in self.precompute:
            self.precompute[key] = {}

        if not save_data:
            X = self.__dict__.pop('X')
            Y = self.__dict__.pop('Y')

        self._compress_model()
        pickle.dump(self, open(save_path, 'wb'))
        self._decompress_model()

        # add back model data
        if not save_data:
            self.__dict__['X'] = X
            self.__dict__['Y'] = Y
