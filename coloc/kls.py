import numpy as np
from scipy.special import loggamma, digamma

def unit_normal_kl(mu_q, var_q):
    """
    KL (N(mu, var) || N(0, 1))
    """
    KL = 0.5 * (var_q + mu_q ** 2 - np.log(var_q) - 1)
    return KL

def normal_kl(mu_q, var_q, mu_p, var_p):
    KL = 0.5 * (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + 2 * np.log(np.sqrt(var_p) / np.sqrt(var_q)))
    return KL

def gamma_kl(a_q, b_q, a_p, b_p):
    KL = (a_q - a_p) * digamma(a_q) \
        - loggamma(a_q) + loggamma(a_p) \
        + a_p * (np.log(b_q) - np.log(b_p)) + a_q * (b_p - b_q) / b_q
    return KL

def normal_entropy(var):
    entropy = (1 / 2) * np.log(2 * np.pi * var) + 1
    return entropy

def categorical_kl(pi_q, pi_p):
    """
    KL(pi_q || pi_p)
    """
    return np.sum(pi_q * (np.log(pi_q + 1e-10) - np.log(pi_p + 1e-10)))

def bernoulli_kl(q, p):
    """
    q and p are probability of success
    """
    return q * np.log(q +1e-10) + (1 - q) * np.log(1 - q + 1e-10) - q * np.log(p + 1e-10) - (1 - q) * np.log(1 - p + 1e-10)
