import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from types import SimpleNamespace
import os
import random
import string
from .misc import *

def linregress(y, X):
    """
    y: m x t expression
    X: m x n genotype
    compute t x n pairwise linear regressions
    reports slopes/standard errors
    """
    diag = np.einsum('ij,ij->i', X.T, X.T)
    betas = y.T @ X / diag
    if np.ndim(y) == 1:
        var = np.var(y[:, None] - betas * X, 0) / diag
    else:
        var = np.array([np.var(y[:, t][:, None] - betas[t] * X, 0) / diag for t in range(y.shape[1])])
    return betas, np.sqrt(var)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self)))
        
class RegressionData(dotdict):
    def __init__(self, X=None, Y=None, Z=None):
        """
        X = genotype  [M, N]
        Y = expression [M, T]
        Z = covariates [M, Q]
        """
        self.x_mean = self.y_mean = self.z_mean = None
        self.X = X
        self.Y = Y
        self.Z = Z
        self.xcorr = None

    def get_summary_stats(self):
        """
        compute pairwise summary stats
        """
        if self.Z is not None:
            self.remove_covariates()
        #self.center_data()
        self.beta, self.se = linregress(self.Y, self.X)

    def get_cafeh_data(self):
        """
        shape data for CAFEHS
        """
        return {
            'LD': np.corrcoef(self.X, rowvar=False),
            'B': self.beta,
            'S': np.sqrt((self.beta**2 / self.X.shape[0]) + self.se**2)
        }

    def get_cafehg_data(self):
        """
        shape data for CAFEHG
        """
        return {
            'X': self.X,
            'Y': self.Y
        }

    def remove_covariates(self):
        """
        get residual of Y ~ Z
        """
        if self.Z is not None:
            self.Y -= self.Z @ (np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T @ self.Y)
            self.Z = None

    def center_data(self):
        """
        center all data
        """
        # for np.array: np.mean(Z, axis=0, keepdims=True)
        # for np.matrix, no keepdims argument
        if self.X is not None and self.x_mean is None:
            self.x_mean = np.mean(self.X, axis=0)
            self.X -= self.x_mean
        if self.Y is not None and self.y_mean is None:
            self.y_mean = np.mean(self.Y, axis=0)
            self.Y -= self.y_mean
        if self.Z is not None and self.z_mean is None:
            self.z_mean = np.mean(self.Z, axis=0)
            self.Z -= self.z_mean

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def thin(data):
    print('thinning data')
    d = int(data.common_snps.size / 1000)
    common_snps = data.common_snps[::d]
    common_snps = common_snps[:1000]

    genotype = data.genotype_gtex.loc[:, common_snps]
    genotype1kG = data.genotype_1kG.loc[:, common_snps]
    B = data.B.loc[:, common_snps]
    S = data.S.loc[:, common_snps]
    V = data.V.loc[:, common_snps]
    n = data.n.loc[:, common_snps]

    X = center_mean_impute(genotype).values
    X1kG = center_mean_impute(genotype1kG).values

    return SimpleNamespace(**{
        'genotype_1kG': genotype1kG,
        'genotype_gtex': genotype,
        'X': X, 'X1kG': X1kG, 'expression': data.expression,
        'B': B, 'S': S, 'V':V, 'n':n, 'common_snps': common_snps,
        'gene': data.gene, 'id': data.gene, 'covariates': data.covariates
    })


def sim_expression(data, n_studys=1, n_causal=None, pve=0.1, active=None, causal_snps=None, true_effects=None, study_variance=None):
    """
    generic function for simulating expression
    returns expression, causal_snps, true effects and study variance
    you can fix any of the returned values by passing values
    anything not specified will be randomly generated
    TODO: specify how to generate active
    """
    n_samples, n_snps = data.X.shape
    n_causal = n_causal or true_effects.shape[1]
    # set random seed with function of sim_id so we can recreate
    if active is None:
        active = 0  # sim actve
    if causal_snps is None:
        causal_snps = np.random.choice(n_snps, n_causal, replace=False)
    if true_effects is None:
        true_effects = np.random.normal(size=(n_studys, n_causal)) * active
    if study_variance is None:
        if np.isscalar(pve):
            pve = np.ones(true_effects.shape[0]) * pve
        study_variance = np.array([
            compute_sigma2(data.X[:, causal_snps], te, pve)
            for te, pve in zip(true_effects, pve)
        ])
    #simulate expression
    expression = (true_effects @ data.X[:, causal_snps].T) + \
        np.random.normal(size=(n_studys, n_samples)) * np.sqrt(study_variance)[:, None]
    expression = pd.DataFrame(expression - expression.mean(1)[:, None])

    return SimpleNamespace(**{
        'expression': expression,
        'causal_snps': causal_snps,
        'true_effects': true_effects,
        'pve': pve,
        'study_variance': study_variance
    })

def simulate_n_causal_variants(data, spec):
    """
    simulation with fixed number of causal variants per study
    """
    causal_per_study = spec.causal_per_study
    n_causal = spec.n_causal
    n_studys = spec.n_studys
    pve = spec.pve
    
    print('generating data')    
    print('\tn_studys={}\tcausal_per_study={}\tn_causal={}\tpve={}'.format(
        n_studys, causal_per_study, n_causal, pve))
    
    n_snps = data.X.shape[1]
    true_effects = np.zeros((n_studys, n_causal))
    for t in range(n_studys):
        causal_in_t = np.random.choice(n_causal, causal_per_study, replace=False)
        true_effects[t, causal_in_t] = np.random.normal(size=causal_in_t.size)
    active = (true_effects != 0)
    return sim_expression(
        data, active=active, true_effects=true_effects, pve=pve)


def simulate_max_n_causal_variants(data, spec):
    """
    simulation with fixed number of causal variants per study
    """
    n_causal = spec.n_causal
    n_studys = spec.n_studys
    max_causal_per_study = spec.max_n_causal
    pve = spec.pve

    print('generating data')
    print(spec.to_dict())
    causal_snps = np.random.choice(data.common_snps.size, n_causal)
    true_effects = []
    for _ in range(n_studys):
        causal_in_study = np.random.choice(max_causal_per_study)
        causal_in_study = np.random.choice(n_causal, causal_in_study, replace=False)
        effects = np.zeros(n_causal)
        effects[causal_in_study] = np.random.normal(size=causal_in_study.size)
        true_effects.append(effects)

    true_effects = np.array(true_effects)
    return sim_expression(
        data, pve=pve, n_causal=n_causal,
        causal_snps=causal_snps, true_effects=true_effects)


def sim_expression_from_model(data, spec):
    """
    Use the parameters of a fit cafeh model to simulate expression
    gene: Use genotype in a 1mb window of tss of gene
    thin: select a 1000 snp subset of variants if True
    sim_id: pass sim id to regenerate an old simulation
    """
    model = load(spec.source_model_path)
    active = model.active.max(0)
    purity = np.array([model.records['purity'][k] for k in range(model.dims['K'])])
    active = (active > 0.5) & (purity > 0.7)
    active[10:] = False
    
    true_effects = (model.weight_means * model.pi[None]).sum(-1) * (model.active > 0.5)
    true_effects = true_effects[:, active]
    # random seed for reproducibility
    print('setting random seed')
    return sim_expression(
        data,
        n_studys=model.study_ids.size,
        n_causal=active.sum(),
        active=(model.active > 0.5)[:, active],
        true_effects=true_effects,
        study_variance = 1 / model.expected_study_precision
    )


def compute_summary_stats(sim, data):
    # generate summary stats
    print('generating summary stats')
    summary_stats = [linregress(y, data.X) for y in sim.expression.values]
    B = pd.DataFrame(np.stack([x[0] for x in summary_stats]), columns=data.common_snps)
    V = pd.DataFrame(np.stack([x[1] for x in summary_stats]), columns=data.common_snps)
    S = pd.DataFrame(np.stack([np.sqrt(x[2]) for x in summary_stats]), columns=data.common_snps)
    return SimpleNamespace(**{
        'B': B, 'S': S, 'V': V
    })


def load_genotype_data(gene, thin=False):
    # Load GTEx and 1kG genotype
    # flip genotype encoding to be consistent with GTEx associations
    print('loading genotypes...')
    genotype = load_gtex_genotype(gene)
    genotype1kG = load_1kG_genotype(gene)

    # filter down to list of snps present in GTEx and 1kG
    print('filtering down to common snps')
    common_snps = get_common_snps(gene)

    if thin:
        d = int(common_snps.size / 1000)
        common_snps = common_snps[::d]
        common_snps = common_snps[:1000]

    genotype = genotype.loc[:, common_snps]
    genotype1kG = genotype1kG.loc[:, common_snps]
    X = center_mean_impute(genotype).values
    X1kG = center_mean_impute(genotype1kG).values

    covariates = pd.read_csv(
        '/work-zfs/abattle4/karl/cosie_analysis/output/GTEx/covariates.csv', sep='\t', index_col=[0, 1])
    covariates = covariates.loc[:, genotype.index.values]

    return SimpleNamespace(**{
        'genotype_1kG': genotype1kG,
        'genotype_gtex': genotype,
        'X': X, 'X1kG': X1kG, 'covariates': covariates,
        'common_snps': common_snps,
        'gene': gene, 'id': gene
    })


def load_sim_data(spec):
    """
    Use the parameters of a fit cafeh model to simulate expression
    gene: Use genotype in a 1mb window of tss of gene
    thin: select a 1000 snp subset of variants if True
    sim_id: pass sim id to regenerate an old simulation
    """
    # load data
    np.random.seed(spec.seed)
    data = load_genotype_data(spec.gene, thin=True)

    if spec.sim_method == 'n_causal_variants':
        sim_method = simulate_n_causal_variants
    if spec.sim_method == 'sim_from_model':
        sim_method = sim_expression_from_model
    if spec.sim_method == 'max_n_causal_variants':
        sim_method = simulate_max_n_causal_variants
    # simulate expression
    if not os.path.isfile(spec.sim_path):
        se = sim_method(data, spec)
        print('saving simulated expression to: {}'.format(spec.sim_path))
        pickle.dump(se, open(spec.sim_path, 'wb'))
    else:
        print('loading simulated expression from: {}'.format(spec.sim_path))
        se = pickle.load(open(spec.sim_path, 'rb'))
    summary_stats = compute_summary_stats(se, data)
    # pickle.dump(sim_params, open('{}/{}.sim.params'.format(sim_dir, sim_id), 'wb'))
    return SimpleNamespace(**{
        'summary_stats': summary_stats,
        'simulation': se,
        'data': data,
        'spec': spec
    })