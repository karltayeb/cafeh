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

    def __init__(self, LD, YX, yy, n_samples, K, U=None, prior_pi=None, snp_ids=None, tissue_ids=None, sample_ids=None, tolerance=1e-5):
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
        self.LD = LD
        self.YX = YX
        self.yy = yy
        self.n_samples = n_samples

        if U is None:
            u, s, _ = np.linalg.svd(LD, hermitian=True)
            m = (~np.isclose(s, 0)).sum()
            U = u[:, :m] * s[:m]

        self.U = U

        # set priors
        T, N = YX.shape
        self.dims = {'N': N, 'T': T, 'K': K}

        self.tissue_ids = tissue_ids if (tissue_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)

        self.prior_pi = prior_pi  if (prior_pi is not None) else np.ones(N) / N

        # initialize latent vars
        self.weight_means = np.zeros((T, K, N))
        self.weight_vars = np.ones((T, K, N))
        self.pi = np.ones((K, N)) / N

        # hyper-parameters
        self.a = 1e-10
        self.b = 1e-10

        self.weight_precision_a = np.ones((T, K))
        self.weight_precision_b = np.ones((T, K))

        self.c = 1e-10
        self.d = 1e-10

        self.tissue_precision_a = np.ones(T)
        self.tissue_precision_b = np.ones(T)

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0

        diags = {t: np.ones(N) * self.n_samples[t] for t in range(T)}

        self.precompute = {
            'Hw': {},
            'Ew2': {},
            'first_moments': {},
            'diags': diags,
            'rX': {}
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


    def rX(self, k):
        expected_effects = self.expected_effects
        if k is not None:
            expected_effects -= self.weight_means[:, k] * self.pi[k][None]
        rX = self.YX - ((expected_effects * self.n_samples[:, None]) @ self.U) @ self.U.T
        return rX

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

    def _get_diag(self, tissue):
        """
        get diag(X^T X) for a given tissue
        differs for tissues because of missingness in Y
        """
        return self.precompute['diags'][tissue]

    def compute_residual(self, k=None):
        """
        computes expected residual
        """
        pass

    def _compute_ERSS(self):
        """
        compute ERSS using XY and XX
        """
        ERSS = np.zeros(self.dims['T'])
        for t in range(self.dims['T']):
            diag = self._get_diag(t)

            pt1 = np.sum((self.weight_means[t] ** 2 + self.weight_vars[t]) * self.pi * diag)
            mu_pi = self.weight_means[t] * self.pi

            # mu_pi @ self.XX @ mu_pi
            pt2 = mu_pi @ (self.U * np.sqrt(diag)[:, None])
            pt2 = np.inner(pt2, pt2)

            ERSS[t] = self.yy[t]
            ERSS[t] += -2 * np.inner(self.YX[t], mu_pi.sum(0))
            ERSS[t] += pt1 + np.sum(pt2) - np.sum(np.diag(pt2))
        return ERSS

    def _update_pi_component(self, k, residual=None):
        """
        update pi for a component
        """
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])
        E_ln_alpha = digamma(self.weight_precision_a[:, k]) \
            - np.log(self.weight_precision_b[:, k])
        E_alpha = self.expected_weight_precision[:, k]
        E_w2 = self.compute_Ew2(k)

        # E[ln p(y | w, z, alpha , tau)]
        tmp1 = -2 * self.rX(k) * self.weight_means[:, k] \
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
        diag = np.array([self._get_diag(t) for t in range(self.dims['T'])])

        precision = diag * self.expected_tissue_precision[:, None] \
            + self.expected_weight_precision[:, k][:, None]
        variance = 1 / precision  # [T, N]
        mean = (variance * self.expected_tissue_precision[:, None]) \
            * self.rX(k)

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

    def update_tissue_variance(self):
        """
        update tau, controls tissue specific variance
        """
        ERSS = self._compute_ERSS()
        self.tissue_precision_a = self.c + self.n_samples / 2
        self.tissue_precision_b = self.d + ERSS / 2

    def fit(self, max_iter=1000, verbose=False, components=None, update_weights=True, update_pi=True, update_variance=True, ARD_weights=False, update_covariate_weights=True):
        """
        loop through updates until convergence
        """
        init_time = time.time()
        if components is None:
            components = np.arange(self.dims['K'])

        for i in range(max_iter):
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

    def compute_elbo(self):
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
            expected_conditional += \
                - 0.5 * self.n_samples[tissue] * np.log(2 * np.pi) \
                + 0.5 * self.n_samples[tissue] * E_ln_tau[tissue] \
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
        return self.LD[snps][:, snps]

    def save(self, output_dir, model_name, save_data=False):
        """
        save the model
        """
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]

        if not save_data:
            XX = self.__dict__.pop('XX')
            YX = self.__dict__.pop('YX')
            yy = self.__dict__.pop('yy')

        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))
        if not save_data:
            self.__dict__['XX'] = XX
            self.__dict__['YX'] = YX
            self.__dict__['yy'] = yy

