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
from .misc import load_gene_data, linregress, load

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def sim_expression(data, n_tissues=1, n_causal=0, pve=0, active=None, causal_snps=None, true_effects=None, tissue_variance=None):
    """
    generic function for simulating expression
    returns expression, causal_snps, true effects and tissue variance
    you can fix any of the returned values by passing values
    anything not specified will be randomly generated
    TODO: specify how to generate active
    """
    n_samples, n_snps = data.X.shape
    # set random seed with function of sim_id so we can recreate
    if active is None:
        active = 0  # sim actve
    if causal_snps is None:
        causal_snps = np.random.choice(n_snps, n_causal, replace=False)
    if true_effects is None:
        true_effects = np.random.normal(size=(n_tissues, n_causal)) * active
    if tissue_variance is None:
        tissue_variance = np.array([
            compute_sigma2(data.X[:, causal_snps], te, pve) for te in true_effects
        ])
    #simulate expression
    expression = (true_effects @ data.X[:, causal_snps].T) + \
        np.random.normal(size=(n_tissues, n_samples)) * np.sqrt(tissue_variance)[:, None]
    expression = pd.DataFrame(expression - expression.mean(1)[:, None])

    return SimpleNamespace(**{
        'expression': expression,
        'causal_snps': causal_snps,
        'true_effects': true_effects,
        'tissue_variance': tissue_variance
    })

def sim_expression_from_model(data, model, sim_id, seed):
    """
    Use the parameters of a fit cafeh model to simulate expression
    gene: Use genotype in a 1mb window of tss of gene
    thin: select a 1000 snp subset of variants if True
    sim_id: pass sim id to regenerate an old simulation
    """
    # set random seed with function of sim_id so we can recreate
    np.random.seed(abs(hash(sim_id)) % 100000000)
    active = model.active.max(0) > 0.5
    active[10:] = False

    # random seed for reproducibility
    np.random.seed(seed)
    return sim_expression(
        data,
        n_tissues=data.expression.shape[0],
        n_causal=active.sum(),
        active=(model.active > 0.5)[:, active],
        tissue_variance = 1 / model.expected_tissue_precision
    )


def load_sim_from_model_data(gene, sim_spec):
    """
    Use the parameters of a fit cafeh model to simulate expression
    gene: Use genotype in a 1mb window of tss of gene
    thin: select a 1000 snp subset of variants if True
    sim_id: pass sim id to regenerate an old simulation
    """
    # load model and look up sim_id from sim_spec
    spec = sim_spec[sim_spec.gene == gene].iloc[0]
    gss = load(spec.source_model_path)

    # load data
    data = load_gene_data(gene, thin=True)

    # simulate expression
    if not os.path.isfile(spec.sim_path):
        se = sim_expression_from_model(data, gss, spec.sim_id, spec.seed)
        print('saving simulated expression to: {}'.format(spec.sim_path))
        pickle.dump(spec.sim_path)
    else:
        print('loading simulated expression to: {}'.format(spec.sim_path))
        se = pickle.load(open(spec.sim_path, 'rb'))

    # generate summary stats
    print('generating summary stats')
    summary_stats = [linregress(y, data.X) for y in se.expression.values]
    B = pd.DataFrame(np.stack([x[0] for x in summary_stats]), columns=data.common_snps)
    V = pd.DataFrame(np.stack([x[1] for x in summary_stats]), columns=data.common_snps)
    S = pd.DataFrame(np.stack([np.sqrt(x[2]) for x in summary_stats]), columns=data.common_snps)

    # pickle.dump(sim_params, open('{}/{}.sim.params'.format(sim_dir, sim_id), 'wb'))
    return SimpleNamespace(**{
        'B': B, 'S': S, 'V': V,
        'expression': se.expression,
        'genotype_1kG': data.genotype_1kG,
        'genotype_gtex': data.genotype_gtex,
        'X': data.X, 'X1kG': data.X1kG, 'common_snps': data.common_snps,
        'covariates': None, 'gene': data.gene,
        'true_effects': se.true_effects,
        'causal_snps': se.causal_snps, 'tissue_variance': se.tissue_variance,
        'sim_id': spec.sim_id, 'id': spec.sim_id})
