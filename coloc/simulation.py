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
from .misc import load_gene_data, linregress

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def sim_from_model(gene, model, thin=False, sim_id=None):
    """
    Use the parameters of a fit cafeh model to simulate expression
    gene: Use genotype in a 1mb window of tss of gene
    thin: select a 1000 snp subset of variants if True
    sim_id: pass sim id to regenerate an old simulation
    """
    if sim_id is None:
        sim_id = randomString(5)

    # set random seed with function of sim_id so we can recreate
    np.random.seed(abs(hash(sim_id)) % 100000000)

    data = load_gene_data(gene, thin=True)

    active = model.active.max(0) > 0.5
    causal_snps = np.random.choice(1000, active.sum(), replace=False)
    true_effects = (model.get_expected_weights() * (model.active > 0.5))[:, active]
    tissue_variance = 1 / model.expected_tissue_precision

    #simulate expression
    expression = (true_effects @ data.X[:, causal_snps].T) + \
        np.random.normal(size=(data.expression.shape)) * np.sqrt(tissue_variance)[:, None]
    expression = pd.DataFrame(expression - expression.mean(1)[:, None])

    summary_stats = [linregress(y, data.X) for y in expression.values]
    B = pd.DataFrame(np.stack([x[0] for x in summary_stats]), columns=data.common_snps)
    V = pd.DataFrame(np.stack([x[1] for x in summary_stats]), columns=data.common_snps)
    S = pd.DataFrame(np.stack([np.sqrt(x[2]) for x in summary_stats]), columns=data.common_snps)
    
    # pickle.dump(sim_params, open('{}/{}.sim.params'.format(sim_dir, sim_id), 'wb'))
    return SimpleNamespace(**{
        'B': B, 'S': S, 'V': V,
        'expression': expression,
        'genotype_1kG': data.genotype_1kG,
        'genotype_gtex': data.genotype_gtex,
        'X': data.X, 'X1kG': data.X1kG, 'common_snps': data.common_snps,
        'covariates': None, 'gene': data.gene,
        'true_effects': true_effects,
        'causal_snps': causal_snps, 'tissue_variance': tissue_variance,
        'sim_id': sim_id, 'id': sim_id
    })