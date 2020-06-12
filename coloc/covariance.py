import numpy as np
from sklearn import covariance

def cov2corr(X):
    """
    scale covariance matrix to correlaton matrix
    """
    diag = np.sqrt(np.diag(X))
    return (1/diag[:, None]) * X * (1/diag[None])

sample_ld = lambda data: np.corrcoef(data.X.T)
refernce_ld = lambda data: np.corrcoef(data.X1kG.T)
z_ld = lambda data: np.corrcoef(data.B.T.values / np.sqrt(data.V.T.values))
ledoit_wolf_sample_ld = lambda data: cov2corr(covariance.ledoit_wolf(data.X)[0])
ledoit_wolf_reference_ld = lambda data: cov2corr(covariance.ledoit_wolf(data.X1kG)[0])
ledoit_wolf_z_ld = lambda data: cov2corr(covariance.ledoit_wolf(data.B.values / np.sqrt(data.V.values))[0])
z3_ld = lambda data: z_ld(data)**3

def ref_z_ld(data, alpha=None):
    """
    mix reference ld and zscore ld
    """
    if alpha is None:
        alpha = data.X1kG.shape[0] / (data.X1kG.shape[0] + data.B.shape[0])
    return alpha * refernce_ld(data) + (1 - alpha) * z_ld(data)
