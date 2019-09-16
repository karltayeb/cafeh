import numpy as np
import scipy as sp
import tensorflow as tf
import gpflow as gpflow
import cvxpy

def get_Wt(Yt, data, problem, weights, warm):
    data.value = Yt
    problem.solve(verbose=True, warm_start=warm, solver=cvxpy.OSQP)
    return weights.value

def solve_cholesky(chol, x):
    """
    Solve system Ax=b from cholesky decomposition
    chol @ chol.T = A
    """
    b = sp.linalg.solve_triangular(
        chol.T, sp.linalg.solve_triangular(chol, x, lower=True), lower=False)
    return b

def update_W(Y, gmu, gvar, Sigma, penalty):
    """
    quad terms summed over tissues
    trace term computed only along diagonal -- assumes unit variance
    l1 penalty on W
    """
    n, q = gmu.shape

    # set up parameters/variables
    mean = cvxpy.Parameter((n, q))
    mean.value = gmu

    traces = cvxpy.Parameter((q), nonneg=True)
    chol = np.linalg.cholesky(Sigma + np.eye(n)*1e-6)
    traces.value = np.array([np.trace(solve_cholesky(chol, gv)) for gv in gvar])

    data = cvxpy.Parameter((n))
    weights = cvxpy.Variable((q))
    penalty = cvxpy.Constant(penalty)

    # set up problem
    error = data - mean @ weights.T
    quad = cvxpy.matrix_frac(error, Sigma + np.eye(n)*1e-6)
    trace = cvxpy.sum(traces * cvxpy.square(weights))
    norm = cvxpy.norm1(weights)

    expression = quad + trace + penalty * norm
    constraints = [weights >= 0]
    problem = cvxpy.Problem(cvxpy.Minimize(expression), constraints)

    W = np.array([get_Wt(Yt, data, problem, weights, i > 0) for i, Yt in enumerate(Y.T)])
    return W

def update_W_map(Y, gmu, gvar, Sigma, penalty):
    """
    quad terms summed over tissues
    trace term computed only along diagonal -- assumes unit variance
    l1 penalty on W
    """
    n, q = gmu.shape

    # set up parameters/variables
    mean = cvxpy.Parameter((n, q))
    mean.value = gmu

    traces = cvxpy.Parameter((q), nonneg=True)
    chol = np.linalg.cholesky(Sigma + np.eye(n)*1e-6)
    traces.value = np.array([np.trace(solve_cholesky(chol, gv)) for gv in gvar])

    data = cvxpy.Parameter((n))
    weights = cvxpy.Variable((q))
    penalty = cvxpy.Constant(penalty)

    # set up problem
    error = data - mean @ weights.T
    quad = cvxpy.matrix_frac(error, Sigma + np.eye(n)*1e-6)
    #trace = cvxpy.sum(traces * cvxpy.square(weights))
    trace = 0
    norm = cvxpy.norm1(weights)

    expression = quad + trace + penalty * norm
    constraints = [weights >= 0]
    problem = cvxpy.Problem(cvxpy.Minimize(expression), constraints)

    W = np.array([get_Wt(Yt, data, problem, weights, i > 0) for i, Yt in enumerate(Y.T)])
    return W

def update_W_se(Y, gmu, gvar, Sigma, penalty):
    """
    quad terms summed over tissues
    trace term computed only along diagonal -- assumes unit variance
    l1 penalty on W
    """
    n, q = gmu.shape

    # set up parameters/variables
    mean = cvxpy.Parameter((n, q))
    mean.value = gmu

    traces = cvxpy.Parameter((q), nonneg=True)
    chol = np.linalg.cholesky(Sigma + np.eye(n)*1e-6)
    traces.value = np.array([np.trace(solve_cholesky(chol, gv)) for gv in gvar])

    data = cvxpy.Parameter((n))
    weights = cvxpy.Variable((q))
    penalty = cvxpy.Constant(penalty)

    # set up problem
    error = data - mean @ weights.T
    quad = cvxpy.sum_squares(error)
    trace = cvxpy.sum(traces * cvxpy.square(weights))
    norm = cvxpy.norm1(weights)

    expression = quad + trace + penalty * norm
    constraints = [weights >= 0]
    problem = cvxpy.Problem(cvxpy.Minimize(expression), constraints)

    W = np.array([get_Wt(Yt, data, problem, weights, i > 0) for i, Yt in enumerate(Y.T)])
    return W

def _project_precomputed(Kxx, Kxz, Kzz, KxzKzzinv, gmu_z, gvar_z):
    mu = (KxzKzzinv @ gmu_z)
    var = (Kxx - KxzKzzinv @ Kxz.T) + (KxzKzzinv @ gvar_z @ KxzKzzinv.T) # Kxz Kzz_inv Sigma_g Kzz_inv Kzx
    return mu, var

def _make_pseudo_data(W, Y, qgmu, qgvar, m):
    # create pseudo data for update
    # looks like residual after other components taken out
    gmu_del_m = np.delete(qgmu, m, axis=1)
    W_del_m = np.delete(W, m, axis=1)

    f_del_m = (gmu_del_m @ W_del_m.T)  # N x T
    pseudo_data = (W[:, m][None] * (Y - f_del_m)).sum(1)
    return pseudo_data

def _update_component_inducing(W, precompute, Y, chol, qgmu, qgvar, m):
    """
    X, Y training inputs/outputs
    Z inducing points
    qgmu, qgvar variational aproximation of g(X)
    m index of the component to update
    """
    Kxx, Kxz, Kzz, KxzKzzinv = precompute[m]
    n = Kxx.shape[0]
    z = Kzz.shape[0]
    pseudo_data = _make_pseudo_data(W, Y, qgmu, qgvar, m)

    total_weight = np.sum(W[:, m]**2)
    temp = Kxz.T @ (total_weight * solve_cholesky(chol, Kxz)) + Kzz
    Sigma_gm = Kzz @ np.linalg.solve(temp + np.eye(z)*1e-6, Kzz)  # small inversion, quick to solve

    # compute mean
    mu_gm = Sigma_gm @ KxzKzzinv.T @ solve_cholesky(chol, pseudo_data)
    return mu_gm, Sigma_gm

def _update_component(W, precompute, Y, chol, qgmu, qgvar, m):
    """
    X, Y training inputs/outputs
    Z inducing points
    qgmu, qgvar variational aproximation of g(X)
    m index of the component to update
    """
    Kxx, _, _, _ = precompute[m]
    n = Kxx.shape[0]
    pseudo_data = _make_pseudo_data(W, Y, qgmu, qgvar, m)
    total_weight = np.sum(W[:, m]**2)

    temp = total_weight * solve_cholesky(chol, Kxx) + np.eye(n)
    Sigma_gm = np.linalg.solve(temp, Kxx).T  # This is the expensive part

    # compute mean
    mu_gm = Sigma_gm @ solve_cholesky(chol, pseudo_data)
    return mu_gm, Sigma_gm

def compute_gmu_approx_inducing(W, Y, precompute, chol, q_gmu_z, q_gvar_z, niter=1):
    q = len(q_gmu_z)

    # project inducing points to trainind data
    q_gmu = []
    q_gvar = []
    for component in range(q):
        mu, var = _project_precomputed(*precompute[component], q_gmu_z[component], q_gvar_z[component])
        q_gmu.append(mu)
        q_gvar.append(var)
    q_gmu = np.array(q_gmu).T
    q_gvar = np.array(q_gvar)

    for _ in range(niter):
        for component in range(q):
            total_weight = (W[:, component] ** 2).sum()
            if total_weight > 1e-6:
                mu_gm_z, Sigma_gm_z = _update_component_inducing(
                    W, precompute, Y, chol, q_gmu, q_gvar, component)

            # update apprxoimations
            q_gmu_z[component] = mu_gm_z
            q_gvar_z[component] = Sigma_gm_z

            # update projections
            mu, var = _project_precomputed(*precompute[component], mu_gm_z, Sigma_gm_z)
            q_gmu[:, component] = mu
            q_gvar[component] = var

    return q_gmu_z, q_gvar_z, q_gmu, q_gvar

def compute_gmu_approx(W, Y, precompute, chol, q_gmu, q_gvar, niter=1, restart=False):
    n, q = q_gmu.shape
    for _ in range(niter):
        for component in range(q):
            total_weight = (W[:, component] ** 2).sum()
            if total_weight > 1e-6:
                mu_gm, Sigma_gm = _update_component(
                    W, precompute, Y, chol, q_gmu, q_gvar, component)
            else:
                if restart:
                    mu_gm = (Y - q_gmu @ W.T).mean(1)
                else:
                    mu_gm = np.zeros(n)
                Sigma_gm = precompute[component][2]

            # update projections
            q_gmu[:, component] = mu_gm
            q_gvar[component] = Sigma_gm

    return q_gmu, q_gvar

def _make_covariances(kernel, X, Zs):
    precompute = []
    for component in range(len(kernel.kernels)):
        Z = Zs[component]
        Kxx = kernel.kernels[component].compute_K_symm(X)
        Kxz = kernel.kernels[component].compute_K(X, Z)
        Kzz = kernel.kernels[component].compute_K_symm(Z)
        KxzKzzinv = np.linalg.solve(Kzz + np.eye(Kzz.shape[0])*1e-6, Kxz.T).T
        precompute.append((Kxx, Kxz, Kzz, KxzKzzinv))
    return precompute

def update_variational_params_inducing(W, precompute, Y, Sigma, q_gmu_z, q_gvar_z, niter=10):
    """
    Zs, q_gmu_z, q_gvar_z are lists with inducing points, and variational parameters respectively
    """

    chol = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * 1e-6)

    # get mean and variance on training data
    mu_gm_z, Sigma_gm_z, q_gmu, q_gvar = compute_gmu_approx_inducing(
        W, Y, precompute, chol, q_gmu_z, q_gvar_z, niter=niter)

    return mu_gm_z, Sigma_gm_z, q_gmu, q_gvar

def update_variational_params(W, X, Y, Sigma, q_gmu, q_gvar, niter=10, restart=True, precompute=None):
    """
    Zs, q_gmu_z, q_gvar_z are lists with inducing points, and variational parameters respectively
    """

    chol = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * 1e-6)
    # make covariance matrices if we are not passed them
    #if precompute is None:
    #    precompute = _make_covariances(model.kern, X, indices)

    # get mean and variance on training data
    q_gmu, q_gvar = compute_gmu_approx(
        W, Y, precompute, chol, q_gmu, q_gvar, niter=niter, restart=restart)

    return q_gmu, q_gvar

class MultipleLinearKernel:
    def __init__(self, W, Zs, kernel=None):
        self.W = W
        self.kernels = [LocalLinearKernal(Z, kernel) for Z in Zs]

class LocalLinearKernal:
    def __init__(self, Z, kernel='fic'):
        """
        W a set of weights
        Z a set of inducing points
        """
        self.Z = Z
        self.chol = np.linalg.cholesky(Z @ Z.T + np.eye(Z.shape[0])*1e-6)
        self.kernel = kernel

    def compute_K(self, X1, X2=None):
        """
        return covariance between sets of inputs X1 and X2
        """
        if X2 is None:
            X2 = X1

        Q = (X1 @ self.Z.T) @ solve_cholesky(self.chol, (X2 @ self.Z.T).T)
        K = X1 @ X2.T

        if self.kernel == 'linear':
            return K
        elif self.kernel == 'fic':
            return np.isclose(K, 1) * (K - Q) + Q
        elif self.kernel == 'sor':
            return Q
        else:
            return None

    def compute_K_symm(self, X):
        """
        retunr covariance between inputs X
        """
        return self.compute_K(X)
