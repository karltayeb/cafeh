from cafeh.cafeh_genotype import CAFEHGenotype
from cafeh.misc import plot_components
from cafeh.fitting import weight_ard_active_fit_procedure
from cafeh.model_queries import *

def fit_cafeh_genotype(X, y, K=10, init_args={}, fit_args = {}):
    """
    Fit CAFEH using individual level genotype data
    LD: LD matrix
    X: [p, n] matrix of genotypes
    y: [t, n] matrix of observtions
    n: int or [t] number of samples in each phenotype,
        if not provided a large sample approximation is made
    """
    cafehg = CAFEHGenotype(X, y, K=10, p0k=0.1)
    weight_ard_active_fit_procedure(cafehg, verbose=False)
    return cafehg

def fit_cafeh_summary(LD, beta, stderr, n=np.inf, K=10, init_args={}, fit_args = {}):
    """
    Fit cafeh using reference LD, effect sizes, and standard errors
    
    LD: LD matrix
    beta: [t, p] matrix of effect sizes
    stderr: [t, p] matrix of standard error
    n: int or [t] number of samples in each phenotype,
        if not provided a large sample approximation is made
    """
    if not (type(n) == np.ndarray):
        n = np.ones(beta.shape[0]) * n
    S = np.sqrt(beta**2/n[:, None] + stderr**2)
    
    cafehs = CAFEHSummary(LD, B, S, K=K, **init_args)
    weight_ard_active_fit_procedure(cafehs, **fit_args)
    return cafehs

def fit_cafeh_z(LD, z, K=10, init_args={}, fit_args={}):
    """
    Fit CAFEH using reference LD and zscores
    LD: LD matrix
    beta: [t, p] matrix of effect sizes
    stderr: [t, p] matrix of standard error
    n: int or [t] number of samples in each phenotype,
        if not provided a large sample approximation is made
    """
    cafehz = CAFEHSummary(LD, z, np.ones_like(z), K=K, **init_args)
    weight_ard_active_fit_procedure(cafehz, max_iter=100)
    return cafehz