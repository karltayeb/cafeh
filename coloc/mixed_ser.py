import cvxpy
import numpy as np
import scipy as sp

def solve_cholesky(cholA, b):
    return sp.linalg.solve_triangular(
        cholA.T, sp.linalg.solve_triangular(cholA, b, lower=True), lower=False)


def make_problem(N, K, nonneg=False):
    """
    make cvxpy problem that solves the weights for a given tissue
    """
    weights_t = cvxpy.Variable(K)

    _beta_means = cvxpy.Parameter(K)
    _beta_vars = cvxpy.Parameter(K, nonneg=True)
    _pi = cvxpy.Parameter((N, K))
    _penalty = cvxpy.Parameter((), pos=True)

    # _B = cvxpy.Parameter((K, K), PSD=True)  # beta @ Kzx Sigma_inv Kxz beta
    _data = cvxpy.Parameter(N)


    lin = -2 * (_data @ _pi @ cvxpy.diag(_beta_means)) @ weights_t
    quad = cvxpy.square(weights_t) @ (cvxpy.square(_beta_means) + _beta_vars)
    # quad = cvxpy.quad_form(weights_t, _B)
    # trace = cvxpy.square(weights_t) @ _beta_vars
    l1 = cvxpy.norm1(weights_t)

    expression = cvxpy.sum(lin + quad + (2 * _penalty * l1))

    if nonneg:
        constraints = [weights_t >= 0]
    else:
        constraints = []
    problem = cvxpy.Problem(cvxpy.Minimize(expression), constraints)

    param_dict = {
        '_beta_means': _beta_means,
        '_beta_vars': _beta_vars,
        '_pi': _pi,
        '_penalty': _penalty,
        '_data': _data
    }
    return problem, param_dict

def set_params(pi, beta_means, beta_vars, penalty, param_dict):
    #Kxz = Sigma @ pi
    # Sigma_inv_Kxz = np.linalg.solve(Sigma, Kxz)
    # A = Kxz.T @ Sigma_inv_Kxz
    #A = Kxz.T @ pi

    # A = A + np.diag(np.ones_like(beta_means) - np.diag(A))
    # B = np.diag(beta_means) @ A @ np.diag(beta_means)

    param_dict['_beta_means'].value = beta_means
    param_dict['_beta_vars'].value = beta_vars
    param_dict['_pi'].value = pi
    param_dict['_penalty'].value = penalty


def solve_w_tissue(Yt, param_dict, problem):
    param_dict['_data'].value= Yt
    problem.solve()
    return problem.variables()[0].value


def update_pi(X, Y, pi, beta_means, beta_vars, weights, k):
    T, N = Y.shape

    # compute residual
    residual = Y - (beta_means * weights) @ (X @ pi).T

    # remove effect of kth component from residual
    r_k = residual + ((beta_means[k] * weights[:, k])[:, None] * (X @ pi[:, k])[None])
    
    # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
    pi_k = r_k * weights[:, k][:, None] * beta_means[k]
    pi_k = pi_k.sum(0)

    # normalize to probabilities
    pi_k = np.exp(pi_k - pi_k.max() + 10)
    pi_k = pi_k / pi_k.sum()
    pi[:, k] = pi_k


def update_beta(X, Y, pi, beta_means, beta_vars, weights, k):
    # compute residual
    old_beta_means = beta_means.copy()
    old_beta_vars = beta_vars.copy()

    residual = Y - (beta_means * weights) @ (X @ pi).T

    # remove effect of kth component from residual
    r_k = residual + ((beta_means[k] * weights[:, k])[:, None] * (X @ pi[:, k])[None])

    # get relevant pi
    pi_k = pi[:, k]

    # now update beta with new pi
    beta_vars[k] = 1 / (1 + np.sum(weights[:, k]**2))
    beta_means[k] = (beta_vars[k]) * \
        np.inner(pi_k, (r_k * weights[:, k][:, None]).sum(0))

        #(Sigma * pi_k).sum(1) @ solve_cholesky(chol, (r_k * weights[:, k][:, None]).sum(0))

def update_beta2(Y, pi, beta_means, beta_vars, weights, Sigma, dist, k, chol=None):
    # compute residual
    r = Y - (beta_means * weights) @ (Sigma @ pi).T

    # remove effect of kth component from residual
    r_k = r + ((beta_means[k] * weights[:, k])[:, None] * (Sigma @ pi[:, k])[None])

    # get relevant pi
    pi_k = pi[:, k]

    # this old update isnt necessary... Kzx Siginv Kxz = 1 for all z...
    # now update beta with new pi
    var_Kxz = (Sigma * pi_k @ Sigma) - \
        ((Sigma * pi_k).sum(1) * (Sigma * pi_k).sum(1)[:, None])
    beta_vars[k] = 1 / (1 + (weights[:, k]**2).sum() * (1 + np.trace(solve_cholesky(chol, var_Kxz))))
    # beta_vars[k] = 1 / (1 + np.sum(weights[:, k]**2))
    beta_means[k] = (beta_vars[k]) * \
        (Sigma * pi_k).sum(1) @ solve_cholesky(chol, (r_k * weights[:, k][:, None]).sum(0))

def unit_normal_kl(mu_q, var_q):
    """
    KL (N(mu, var) || N(0, 1))
    """
    KL = 0.5 * (var_q + mu_q ** 2 - np.log(var_q) - 1)
    return KL

def categorical_kl(pi_q, pi_p):
    """
    KL(pi_q || pi_p)
    """
    return np.sum(pi_q * (np.log(pi_q + 1e-10) - np.log(pi_p + 1e-10)))


def compute_elbo(X, Y, pi, beta_means, beta_vars, weights, dist, penalty=1):
    """
    compute elbo (up to constant)
    """
    K = pi.shape[1]
    Kzz = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            pi1 = pi[:, i]
            pi2 = pi[:, j]
            Kzz[i, j] = np.sum(np.outer(pi1, pi2) * X)
            
    Kzz = Kzz + np.diag(np.ones(K) - np.diag(Kzz))

    T, N = Y.shape
    K = pi.shape[1]

    # bound = np.sum(dist.logpdf(Yt) for Yt in Y)
    bound = 0
    # likelihood term from data
    for t in range(T):
        weights_t = weights[t]
        bound += np.inner(Y[t], pi @ (weights_t * beta_means))
        bound += -0.5 * np.inner((weights_t * beta_means), Kzz @ (weights_t * beta_means))
        bound += -0.5 * np.sum(weights_t**2 * beta_vars)
        # bound += -0.5 * np.sum(weights_t**2 * (beta_means**2 + beta_vars))

    # KL terms
    for k in range(K):
        bound -= unit_normal_kl(beta_means[k], beta_vars[k])
        bound -= categorical_kl(pi[:, k], np.ones(N)/N)

    # likelihood under exponential/laplace distribution
    bound += -1 * penalty * np.sum(np.abs(weights))
    return bound
