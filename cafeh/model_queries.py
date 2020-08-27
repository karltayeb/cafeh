import numpy as np
from scipy.stats import norm
import pandas as pd
from .kls import unit_normal_kl, normal_kl, categorical_kl

#########################
# MODEL QUERY FUNCTIONS #
#########################
def check_convergence(self):
    if len(self.elbos) > 2:
        delta = np.abs((self.elbos[-1] - self.elbos[-2]) / self.elbos[-2])
        if delta < self.tolerance:
            return True
    else:
        return False

def get_expected_weights(self):
    if self.weight_means.ndim == 2:
        weights = self.weight_means * self.active
    else:
        weights = np.einsum('ijk,kj->ij', self.weight_means * self.active[:, :, None], self.pi.T)
    return weights

def get_top_snp_per_component(self):
    """
    returns snp with highest posterior probability for each component
    returns p(max_snp) for each components
    """
    return self.snp_ids[self.pi.T.argmax(axis=0)], self.pi.T.max(axis=0)

def _get_cs(pi, alpha):
    argsort = np.flip(np.argsort(pi))
    cset_size = (np.cumsum(pi[argsort]) < alpha).sum() + 1
    return argsort[:cset_size]

def get_credible_sets(self, alpha=0.95):
    """
    return cs and purity for each component
    alpha the credible set posterior mass
    """
    credible_sets = {k: _get_cs(self.pi[k], alpha) for k in range(self.dims['K'])}
    purity = {}
    for k in range(self.dims['K']):
        if credible_sets[k].size == 1:
            purity[k] = 1.0
        elif credible_sets[k].size < 100:
            LD = self.get_ld(credible_sets[k])
            purity[k] = np.min(np.abs(LD))
        else:
            cs = np.random.choice(credible_sets[k], 100)
            LD = self.get_ld(cs)
            purity[k] = np.min(np.abs(LD))

    return credible_sets, purity

def get_study_pip(self):
    pip = []
    for t in range(self.dims['T']):
        pip.append(1 - np.exp(np.sum(np.log((1 - self.pi * self.active[t][:, None] + 1e-100)), 0)))
    return pd.DataFrame(np.array(pip), columns=self.snp_ids, index=self.study_ids)

def get_pip(self):
    # probability that each component is active in atleast one tissue
    active = 1 - np.exp(np.sum(np.log(1 - self.active + 1e-100), 0))
    # compute pip
    pip = 1 - np.exp(np.sum(np.log((1 - self.pi * active[:, None] + 1e-100)), 0))
    return pip

def get_component_coloc(self):
    """
    returns K x T x T matrix for probability of colocalization of two studys in a component
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))

    coloc = np.zeros((self.dims['K'], self.dims['T'], self.dims['T']))
    for k in range(self.dims['K']):
        right_signs = np.outer(1 - sign_error[:, k], 1 - sign_error[:, k])
        right_signs[np.arange(self.dims['T']), np.arange(self.dims['T'])] = 1 - sign_error[:, k]
        
        both_active = np.outer(self.active[:, k], self.active[:, k])
        both_active[np.arange(self.dims['T']), np.arange(self.dims['T'])] = self.active[:, k]

        coloc[k] = both_active * right_signs
    return coloc

def get_A_in_B_coloc(self):
    """
    returns pairwise probability that active components in study A
    are a subset of the active components of study B

    ie component k on in A -> k on in B
       component k off in B -> k off in A
    """
    # A in B
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.dims['T'], self.dims['T']))
    for t1 in range(self.dims['T']):
        for t2 in range(self.dims['T']):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # 1 - p(component OFF in t2 and component ON in t1)
            probs[t1, t2] = np.exp(np.sum(np.log((1 - (1-B) * A) + 1e-10)))

        probs += np.eye(self.dims['T']) - np.diag(np.diag(probs))
        
    return pd.DataFrame(probs, index=self.study_ids, columns=self.study_ids)

def get_A_equals_B_coloc(self):
    """
    return pairwise probability that the active components in two studys are identicle
    component k on in A <--> k on in B
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.dims['T'], self.dims['T']))
    for t1 in range(self.dims['T']):
        for t2 in range(self.dims['T']):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # prod(p(component on in both studys) +p(component off in both studys))
            probs[t1, t2] = np.exp(np.sum(np.log(A * B + (1 - A) * (1 - B) + 1e-10)))
        probs += np.eye(self.dims['T']) - np.diag(np.diag(probs))
    return pd.DataFrame(probs, index=self.study_ids, columns=self.study_ids)

def get_A_intersect_B_coloc(self):
    """
    return pairwise probability that at least one component is shared by both studys
    component k on in A AND k on in B for some k
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.dims['T'], self.dims['T']))
    for t1 in range(self.dims['T']):
        for t2 in range(self.dims['T']):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # 1 - p(all components off in both studys)
            probs[t1, t2] = 1 - np.exp(np.sum(np.log((1 - A * B) + 1e-10)))
        probs += np.eye(self.dims['T']) - np.diag(np.diag(probs))
    return pd.DataFrame(probs, index=self.study_ids, columns=self.study_ids)


def compute_component_bayes_factors(self, prior_variance):
    """
    bf = np.zeros_like(self.active)
    weight_var = 1 / (1 + (1 /self.prior_variance))

    for k in range(self.dims['K']):
        r_k = self._compute_residual(k)
        for t in range(self.dims['T']):
            bf[t, k] = r_k[t] @ self.pi.T[:, k] * self.weights[t, k] - 0.5 * (self.weights[t, k]**2 + weight_var)
    """
    z = np.array([self._compute_residual(k=k) @ self.pi.T[:, k] for k in range(self.dims['K'])]).T
    log_bf = -0.5 * np.log(prior_variance + 1) + (z ** 2) / 2 * (prior_variance / (1 + prior_variance))
    return log_bf
