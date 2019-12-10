import numpy as np
from scipy import stats

class SuSiE:
    def __init__(self, X, Y, K, prior_weight_variance, prior_slab_weights, pi_prior):
        """
        Y [M] expresion for tissue, individual
        X [N x M] genotype for snp, individual
            potentially [T x N x M] if tissue specific correction

        prior_weight_variance [T, K]
            prior variance for weight of (tissue, component) loading
        prior_slab_weights [K]
            prior probability of sapling from slab in component k
        pi_prior: prior for multinomial,
            probability of sampling a snps as the active feature
        """

        # set data
        self.X = X
        self.Y = Y

        # set priors
        M = Y.size
        N = X.shape[0]
        self.dims = {'N': N, 'M': M, 'K': K}

        self.variance = 1.0
        self.prior_variance = np.ones(K)

        self.pi = np.ones((K, N)) / N
        self.weight_means = np.zeros((K, N))
        self.weight_vars = np.ones((K, N))

        self.elbos = []
        self.tolerance = 1e-5

    def _compute_prediction(self, k=None):
        prediction = ((self.weight_means * self.pi) @ self.X).sum(0)
        if k is not None:
            prediction -= (self.weight_means[k] * self.pi[k]) @ self.X
        return prediction

    def _compute_residual(self, k=None):
        return self.Y - self._compute_prediction(k)

    def _update_prior_variance(self, k):
        pass

    def _update_variance(self):
        ERSS = np.sum(self._compute_residual()**2)
        b = (self.pi * self.weight_means) @ self.X
        ERSS -= np.einsum('ij, ij->i', b, b).sum()
        ERSS += ((self.pi * (self.weight_vars + self.weight_means**2)) @ (self.X**2)).sum()
        return ERSS / self.dims['N']

    def SER(self, k):
        r = self._compute_residual(k)
        p_means = []
        p_vars = []
        BFs = []

        for x in self.X:
            p_mean, p_var, BF = _update(r, x, self.variance, self.prior_variance[k])
            p_means.append(p_mean)
            p_vars.append(p_var)
            BFs.append(BF)

        p_means = np.array(p_means)
        p_vars = np.array(p_vars)
        BFs = np.array(BFs)

        self.pi[k] = BFs / (BFs.sum() - BFs)
        self.weight_means[k] = p_means
        self.weight_vars[k] = p_vars

    def fit(self):
        for component in range(self.dims['K']):
            # EB prior variance
            self._update_prior_variance(component)

            # SER on the component
            self.SER(component)
        self._update_variance()


def _update(y, x, sigma2, sigma02):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    z = slope / std_err
    s2 = std_err**2

    BF = np.sqrt(s2 / (sigma02 + s2)) * np.exp(z**2/2 * sigma02/(sigma02 + s2))
    p_var = 1 / (s2 + sigma02)
    p_mean = p_var / s2 * slope
    return p_mean, p_var, BF
