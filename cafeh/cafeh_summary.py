import numpy as np
from .kls import categorical_kl, bernoulli_kl, normal_entropy, gamma_kl
import os, sys, pickle
from .utils import centered_moment2natural, natural2centered_moment
import pandas as pd
from scipy.special import digamma
import time

class CAFEHSummary:
    """
    CAFEH RSS estimate of CAFEH-G with spike and slab weights
    """
    from .plotting import plot_components, plot_assignment_kl, plot_credible_sets_ld, plot_decomposed_zscores, plot_pips
    from .model_queries import get_credible_sets, get_purity, get_pip, get_study_pip, get_expected_weights, check_convergence

    def __init__(self, LD, B, S, K, p0k=0.1, prior_variance=0.1, snp_ids=None, study_ids=None, tolerance=1e-5):
        """
        blah
        """

        # set data
        self.LD = LD  # N x N
        self.B = B  # T x N z scores
        self.S = S  # T x N S from RSS likelihood
        # set priors
        T, N = B.shape
        self.dims = {'N': N, 'T': T, 'K': K}

        self.study_ids = study_ids if (study_ids is not None) else np.arange(T)
        self.snp_ids = snp_ids if (snp_ids is not None) else np.arange(N)

        self.prior_precision = np.ones((T, K))
        self.prior_pi = np.ones(N) / N
        self.prior_activity = np.ones(K) * p0k
        # initialize latent vars
        self.weight_means = np.zeros((T, K, N))
        self.weight_vars = np.ones((T, K, N))
        self.pi = np.ones((K, N)) / N
        self.active = np.ones((T, K))

        # hyper-parameters
        self.a = 1e-10
        self.b = 1e-10

        # variational paramters for (gamma-distributed) weight precisions
        self.weight_precision_a = np.ones((T, K))
        self.weight_precision_b = np.ones((T, K)) * prior_variance

        self.c = 1e-10
        self.d = 1e-10

        # variational paramters for (beta distributed) variance proportions
        self.study_precision_a = np.ones(T)
        self.study_precision_b = np.ones(T)

        self.elbos = []
        self.tolerance = tolerance
        self.run_time = 0
        self.step_size = 1.0

        self.precompute = {
            'Hw': {},
            'Ew2': {},
            'first_moments': {},
            'masks': {}
        }
        self.records = {}

    def clear_precompute(self):
        self.precompute['Hw'] = {}
        self.precompute['Ew2'] = {}
        self.precompute['first_moments'] = {}

    @property
    def expected_study_precision(self):
        """
        expected precision for study under variational approximation
        """
        # for now we will not mess with this
        return self.study_precision_a / self.study_precision_b

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

    @property
    def expected_log_odds(self):
        """
        computed expected effect size E[zw] [T, N]
        """
        return np.log(self.prior_activity + 1e-10) - np.log(1 - self.prior_activity + 1e-10)

    def compute_Hw(self, component):
        """
        compute entropy of q(w|z, active=1)
        """
        if component not in self.precompute['Hw']:
            v1 = self.weight_vars[:, component]
            self.precompute['Hw'][component] = normal_entropy(v1)
        return self.precompute['Hw'][component]

    def compute_Ew2(self, component):
        """
        compute second moment of q(w|z, active=1)
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
        return self.get_credible_sets()

    @property
    def purity(self):
        """
        return minimum absolute correlation of snps in each credible set
        """
        return self.get_purity()
        
    def compute_first_moment(self, component):
        """
        compute E[S^{-1}LD zws] for a component
        """

        # if its not computed, compute now
        if component not in self.precompute['first_moments']:
            self.precompute['first_moments'][component] = \
                self._compute_first_moment_randomized(component)
        return self.precompute['first_moments'][component]

    def _compute_first_moment(self, component):
        """
        compute first moment
        """
        pi = self.pi[component]
        weight = self.weight_means[:, component]
        active = self.active[:, component]
        moment = []
        mu = pi[None] * weight * active[:, None]
        m = (mu / self.S) @ self.LD * self.S
        return m

    def _compute_first_moment_randomized(self, component, Q=100):
        """
        compute estimate of first moment, sampling from q(z)
        """
        pi = self.pi[component]
        active = self.active[:, component][:, None]
        sample = np.random.choice(a=pi.size, size=Q, p=pi)
        weight = self.weight_means[:, component, sample]
        mu = (weight * active) / self.S[:, sample]
        m = ((mu @ self.LD[sample]) * self.S) / Q
        return m

    def compute_study_constant(self, study):
        """
        compute E[LDSDzw] for a component
        """
        return np.zeros(self.dims['T'])[0]

    def compute_prediction(self, k=None):
        """
        compute expected prediction
        """
        prediction = np.sum([
            self.compute_first_moment(l) for l in range(self.dims['K']) if l != k
        ], axis=0)
        return prediction

    def compute_residual(self, k=None):
        """
        computes expected residual
        """
        prediction = self.compute_prediction(k)
        return self.B - prediction

    def _compute_ERSS(self):
        """
        compute ERSS using XY and XX
        """
        ERSS = np.array([self._compute_ERSS_study(t)
            for t in range(self.dims['T'])])
        return ERSS

    def _compute_ERSS_study(self, t):
        diag = (self.S**-2)[t]  # N
        active = self.active[t][:, None]  # Kx1
        mupi = (self.weight_means[t] * active) * self.pi

        ERSS = self.compute_study_constant(t)
        ERSS += -2 * np.inner(self.B[t] * diag, mupi.sum(0))
        ERSS += self._compute_quad_randomized(t)
        return ERSS

    def _compute_quad_randomized(self, t, Q=1):
        sample = np.array([np.random.choice(
            self.pi[0].size, Q, p=self.pi[k]) for k in range(self.dims['K'])]).T
        diag = self.S[t]**-2
        active = self.active[t]
        total = []
        for s in sample:
            beta_s = (active * self.weight_means[t, np.arange(self.dims['K']), s]) / self.S[t, s]
            var_beta = (self.weight_vars[t, np.arange(self.dims['K']), s] * active)
            total.append((beta_s @ self.LD[s][:, s] @ beta_s)
                         + (diag[s] * var_beta).sum())
        return np.mean(total)

    def _compute_quad(self, t):
        diag = (self.S**-2)[t] # N
        active = self.active[t][:, None]  # Kx1
        mupi = (self.weight_means[t] * active) * self.pi

        Ew2 = (self.weight_means[t]**2 + self.weight_vars[t])  #KxN
        m2pid = np.sum(active * Ew2 * diag * self.pi)
        mpSpm = (mupi/self.S[t]) @ self.LD @ (mupi/self.S[t]).T
        total = m2pid.sum() + np.sum(mpSpm) - np.sum(np.diag(mpSpm))
        return total

    def _update_pi_component(self, k, residual=None):
        """
        update pi for a component
        """
        diag = self.S**-2
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual

        E_ln_alpha = digamma(self.weight_precision_a[:, k]) \
            - np.log(self.weight_precision_b[:, k])
        E_alpha = self.expected_weight_precision[:, k][:, None]
        Ew2 = self.compute_Ew2(k)
        active = self.active[:, k][:, None]

        # E[ln p(y | w, s, z, tau)]
        tmp1 = -2 * r_k * self.weight_means[:, k] + Ew2
        tmp1 = -0.5 * self.expected_study_precision[:, None] \
            * tmp1 * diag * active

        # E[ln p(w | alpha)] + H(q(w))
        Ew2 = Ew2 * active + (1 / E_alpha) * (1 - active)
        entropy = self.compute_Hw(k)
        entropy = entropy * active + normal_entropy(1 / E_alpha) * (1 - active)
        lik = (- 0.5 * E_alpha * Ew2)  # [T, N]        
        
        pi_k = (tmp1 + lik + entropy)
        pi_k = pi_k.sum(0)
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
        second_moment = Ew2 @ self.pi[k]
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

    def _update_weight_component(self, k, ARD=True, residual=None):
        """
        update weights for a component
        """
        diag = self.S**-2
        if residual is None:
            r_k = self.compute_residual(k)
        else:
            r_k = residual

        precision = diag * self.expected_study_precision[:, None] \
            + self.expected_weight_precision[:, k][:, None]
        variance = 1 / precision  # [T, N]
        mean = (variance * self.expected_study_precision[:, None]) \
            * (r_k * diag)

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
        diag = self.S**-2  # T x N
        r_k = self.compute_residual(k)
        p_k = self.pi[k][None] * self.weight_means[:, k]
        tmp1 = -2 * np.einsum('ij,ij->i', r_k * diag, p_k) \
            + (self.compute_Ew2(k) * diag) @ self.pi[k]
        tmp1 = -0.5 * self.expected_study_precision * tmp1

        tmp2 = -0.5 * self.expected_weight_precision[:, k] \
            * (self.compute_Ew2(k) @ self.pi[k]) \
            + normal_entropy(self.weight_vars[:, k]) @ self.pi[k]

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
        pass

    def fit(self, max_iter=1000, verbose=False, components=None, **kwargs):
        """
        loop through updates until convergence
        """
        init_time = time.time()
        if components is None:
            components = np.arange(self.dims['K'])

        for i in range(max_iter):
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
            #if update_variance:
            #    self.update_study_variance()

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

        E_ln_tau = digamma(self.study_precision_a) - np.log(self.study_precision_b)
        E_tau = self.expected_study_precision

        ERSS = self._compute_ERSS()
        for study in range(self.dims['T']):
            expected_conditional += \
                - 0.5 * self.dims['N'] * np.log(2 * np.pi) \
                + 0.5 * self.dims['N'] * E_ln_tau[study] \
                - 0.5 * E_tau[study] * ERSS[study]

        Ew2 = np.array([self.compute_Ew2(k) for k in range(self.dims['K'])])
        Ew2 = np.einsum('jik,jk->ij', Ew2, self.pi)
        Ew2 = Ew2 * self.active + (1 - self.active) / E_alpha

        Hw = np.array([self.compute_Hw(k) for k in range(self.dims['K'])])
        entropy = np.einsum('jik,jk->ij', Hw, self.pi)
        entropy = entropy * self.active + \
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
        return self.LD[snps][:, snps]

    def compute_weight_variance(self, k):
        """
        compute weight variance parameters, need S and fit parameters
        """
        diag = self.S**-2
        precision = diag * self.expected_study_precision[:, None] \
            + self.expected_weight_precision[:, k][:, None]
        return 1/precision

    def _get_compress_mask(self):
        """
        get snps to retain for each component
        """
        return self.pi > 3/self.pi.shape[1]

    def _compress_model(self):
        """
        compress weight means and weight vars
        save a copy of S
        """
        mask = self._get_compress_mask()
        self.weight_means = self.weight_means[:, mask]
        self.weight_vars = None

    def _decompress_model(self):
        """
        unpack model for use
        """
        mask = self._get_compress_mask()
        weight_means = np.zeros((self.dims['T'], self.dims['K'], self.dims['N']))
        weight_means[:, mask] = self.weight_means

        weight_vars = np.zeros((self.dims['T'], self.dims['K'], self.dims['N']))
        for k in range(self.dims['K']):
            weight_vars[:, k] = self.compute_weight_variance(k)
        self.weight_means = weight_means
        self.weight_vars = weight_vars

    def save(self, save_path, save_ld=False, save_data=False):
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

        if not save_ld:
            LD = self.__dict__.pop('LD')

        if not save_data:
            B = self.__dict__.pop('B')

        self._compress_model()
        pickle.dump(self, open(save_path, 'wb'))

        # add back model data
        if not save_ld:
            self.__dict__['LD'] = LD

        if not save_data:
            self.__dict__['B'] = B
        self._decompress_model()

