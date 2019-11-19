import numpy as np

def unit_normal_kl(mu_q, var_q):
    """
    KL (N(mu, var) || N(0, 1))
    """
    KL = 0.5 * (var_q + mu_q ** 2 - np.log(var_q) - 1)
    return KL

def normal_kl(mu_q, var_q, mu_p, var_p):
    KL = 0.5 * (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + 2 * np.log(np.sqrt(var_p) / np.sqrt(var_q)))
    return KL

def categorical_kl(pi_q, pi_p):
    """
    KL(pi_q || pi_p)
    """
    return np.sum(pi_q * (np.log(pi_q + 1e-10) - np.log(pi_p + 1e-10)))
