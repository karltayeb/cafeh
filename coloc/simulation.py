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
from .misc import load_gene_data, linregress, load, compute_sigma2, center_mean_impute

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


def sim_expression(data, n_tissues=1, n_causal=None, pve=0.1,
                   active=None, causal_snps=None, true_effects=None, tissue_variance=None):
    """
    generic function for simulating expression
    returns expression, causal_snps, true effects and tissue variance
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
        true_effects = np.random.normal(size=(n_tissues, n_causal)) * active
    if tissue_variance is None:
        if np.isscalar(pve):
            pve = np.ones(true_effects.shape[0]) * pve
        tissue_variance = np.array([
            compute_sigma2(data.X[:, causal_snps], te, pve)
            for te, pve in zip(true_effects, pve)
        ])
    #simulate expression
    expression = (true_effects @ data.X[:, causal_snps].T) + \
        np.random.normal(size=(n_tissues, n_samples)) * np.sqrt(tissue_variance)[:, None]
    expression = pd.DataFrame(expression - expression.mean(1)[:, None])

    return SimpleNamespace(**{
        'expression': expression,
        'causal_snps': causal_snps,
        'true_effects': true_effects,
        'pve': pve,
        'tissue_variance': tissue_variance
    })

def simulate_n_causal_variants(data, spec):
    """
    simulation with fixed number of causal variants per tissue
    """
    causal_per_tissue = spec.causal_per_tissue
    n_causal = spec.n_causal
    n_tissues = spec.n_tissues
    pve = spec.pve
    
    print('generating data')    
    print('\tn_tissues={}\tcausal_per_tissue={}\tn_causal={}\tpve={}'.format(
        n_tissues, causal_per_tissue, n_causal, pve))
    
    n_snps = data.X.shape[1]
    true_effects = np.zeros((n_tissues, n_causal))
    for t in range(n_tissues):
        causal_in_t = np.random.choice(n_causal, causal_per_tissue, replace=False)
        true_effects[t, causal_in_t] = np.random.normal(size=causal_in_t.size)
    active = (true_effects != 0)
    return sim_expression(
        data, active=active, true_effects=true_effects, pve=pve)

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
        n_tissues=model.tissue_ids.size,
        n_causal=active.sum(),
        active=(model.active > 0.5)[:, active],
        true_effects=true_effects,
        tissue_variance = 1 / model.expected_tissue_precision
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

    # simulate expression
    if not os.path.isfile(spec.sim_path):
        se = sim_method(data, spec)
        print('saving simulated expression to: {}'.format(spec.sim_path))
        #pickle.dump(se, open(spec.sim_path, 'wb'))
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