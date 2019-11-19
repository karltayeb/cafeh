import numpy as np
from scipy.stats import norm
import pandas as pd
from .kls import unit_normal_kl, normal_kl, categorical_kl

#########################
# MODEL QUERY FUNCTIONS #
#########################

def get_top_snp_per_component(self):
    """
    returns snp with highest posterior probability for each component
    returns p(max_snp) for each components
    """
    return self.snp_ids[self.pi.argmax(axis=0)], self.pi.max(axis=0)

def get_credible_sets(self, alpha=0.9, thresh=0.5):
    """
    return snps for active components
    """
    confidence_sets = {}
    active = self.active.max(0) > thresh
    for k in np.arange(self.K)[active]:
        cset_size = (np.cumsum(np.flip(np.sort(self.pi[:, k]))) < alpha).sum() + 1
        cset = np.flip(np.argsort(self.pi[:, k])[-cset_size:])
        confidence_sets[k] = self.snp_ids[cset]
    return confidence_sets

def get_pip(self):
    """
    return posterior inclusion probability for each tissue, SNP pair
    PIP is the probability that a SNP is the causal snp in at least one component
    """
    pip = np.zeros((self.N, self.T))
    for t in range(self.T):
        for n in range(self.N):
            pip[n, t] = 1 - np.exp(np.sum([np.log(1 - self.pi[n, k] * self.active[t, k]) for k in range(self.K)]))

    return pd.DataFrame(pip, index=self.snp_ids, columns=self.tissue_ids)

def get_component_coloc(self):
    """
    returns K x T x T matrix for probability of colocalization of two tissues in a component
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))

    coloc = np.zeros((self.K, self.T, self.T))
    for k in range(self.K):
        right_signs = np.outer(1 - sign_error[:, k], 1 - sign_error[:, k])
        right_signs[np.arange(self.T), np.arange(self.T)] = 1 - sign_error[:, k]
        
        both_active = np.outer(self.active[:, k], self.active[:, k])
        both_active[np.arange(self.T), np.arange(self.T)] = self.active[:, k]

        coloc[k] = both_active * right_signs
    return coloc

def get_A_in_B_coloc(self):
    """
    returns pairwise probability that active components in tissue A
    are a subset of the active components of tissue B

    ie component k on in A -> k on in B
       component k off in B -> k off in A
    """
    # A in B
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.T, self.T))
    for t1 in range(self.T):
        for t2 in range(self.T):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # 1 - p(component OFF in t2 and component ON in t1)
            probs[t1, t2] = np.exp(np.sum(np.log((1 - (1-B) * A) + 1e-10)))

        probs += np.eye(self.T) - np.diag(np.diag(probs))
        
    return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)

def get_A_equals_B_coloc(self):
    """
    return pairwise probability that the active components in two tissues are identicle
    component k on in A <--> k on in B
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.T, self.T))
    for t1 in range(self.T):
        for t2 in range(self.T):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # prod(p(component on in both tissues) +p(component off in both tissues))
            probs[t1, t2] = np.exp(np.sum(np.log(A * B + (1 - A) * (1 - B) + 1e-10)))
        probs += np.eye(self.T) - np.diag(np.diag(probs))
    return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)

def get_A_intersect_B_coloc(self):
    """
    return pairwise probability that at least one component is shared by both tissues
    component k on in A AND k on in B for some k
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))
    probs = np.zeros((self.T, self.T))
    for t1 in range(self.T):
        for t2 in range(self.T):
            A = self.active[t1] * (1 - sign_error[t1])
            B = self.active[t2] * (1 - sign_error[t2])
            # 1 - p(all components off in both tissues)
            probs[t1, t2] = 1 - np.exp(np.sum(np.log((1 - A * B) + 1e-10)))
        probs += np.eye(self.T) - np.diag(np.diag(probs))
    return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)


def compute_component_bayes_factors(self, prior_variance):
    """
    bf = np.zeros_like(self.active)
    weight_var = 1 / (1 + (1 /self.prior_variance))

    for k in range(self.K):
        r_k = self._compute_residual(k)
        for t in range(self.T):
            bf[t, k] = r_k[t] @ self.pi[:, k] * self.weights[t, k] - 0.5 * (self.weights[t, k]**2 + weight_var)
    """
    z = np.array([self._compute_residual(k=k) @ self.pi[:, k] for k in range(self.K)]).T
    log_bf = -0.5 * np.log(prior_variance + 1) + (z ** 2) / 2 * (prior_variance / (1 + prior_variance))
    return log_bf

