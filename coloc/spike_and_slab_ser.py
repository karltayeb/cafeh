import numpy as np

def update_ss_weights(X, Y, weights, active, pi, prior_activity, prior_variance=1.0):
    """
    X is LD/Covariance Matrix
    Y is T x N
    weights  T x K matrix of weight parameters
    active T x K active[t, k] = logp(s_tk = 1)
    prior_activitiy
    """
    old_weights = weights.copy()
    old_active = active.copy()

    T, K = weights.shape
    W = weights * np.exp(active)
    weight_var = prior_variance / (1 + prior_variance)
    for k in range(K):
        # get expected weights
        W = weights * np.exp(active)

        # compute residual
        residual = Y - (W) @ (X @ pi).T

        # remove effect of kth component from residual
        r_k = residual + (W[:, k])[:, None] * (X @ pi[:, k])[None]

        # update p(w | s = 1)
        weights[:, k] = (weight_var) * r_k @ pi[:, k]

        # now update p(s = 1)
        on = r_k @ pi[:, k] * weights[:, k] \
            - 0.5 * (weights[:, k]**2 + weight_var) + np.log(prior_activity[k])
        # on = on - on.max()

        normalizer = np.log(np.exp(on) + (1-prior_activity[k]))
        active[:, k] = (on - normalizer)

    weight_diff = np.abs(old_weights - weights).max()
    active_diff = np.abs(np.exp(old_active) - np.exp(active)).max()
    return np.array([weight_diff, active_diff]).max()

def update_pi(X, Y, weights, active, pi):
    old_pi = pi.copy()
    T, N = Y.shape
    K = pi.shape[1]

    W = weights * np.exp(active)
    active_components = np.exp(active).max(0) > 1e-2

    for k in np.arange(K)[active_components]:
        # compute residual
        residual = Y - W @ (X @ pi).T

        # remove effect of kth component from residual
        r_k = residual + (W[:, k])[:, None] * (X @ pi[:, k])[None]

        # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
        pi_k = r_k * W[:, k][:, None]
        pi_k = pi_k.sum(0)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max() + 10)
        pi_k = pi_k / pi_k.sum()

        # if nans dont update
        # this can happen if pi_k is 0 or constant
        # should revert to prior but that might be disadvantageous
        #if ~np.any(np.isnan(pi_k)):
        pi[:, k] = pi_k

    for k in np.arange(K)[~active_components]:
        pi[:, k] = np.ones(N)/N

    pi_diff = np.abs(pi - old_pi).max()
    return pi_diff

def elbo(X, Y, weights, active, pi, prior_activity, prior_variance):
    pass
