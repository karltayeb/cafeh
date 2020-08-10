import numpy as np
import pandas as pd
import json
import pickle
from sklearn.utils.extmath import randomized_svd
from types import SimpleNamespace
import glob
import os

from coloc.misc import *
from coloc.covariance import *
from coloc.simulation import *

from coloc.independent_model_ss import IndependentFactorSER as GSS
from coloc.cafeh_ss import CAFEH as CSS
from sklearn import covariance

import seaborn as sns
import matplotlib.pyplot as plt

import random
import string
from collections import defaultdict
import multiprocessing
import itertools

def get_component_activity(model, active_thresh=0.5, purity_thresh=0.7):
    #component_activity = model.active.max(0) > active_thresh
    component_activity = (1 - np.exp(np.sum(np.log(1 - model.active + 1e-10), 0))) > 0.5
    component_purity = np.array(
        [model.records['purity'][k] > purity_thresh for k in range(model.dims['K'])])
    return component_purity & component_activity


def all_credible_snps(model, active_thresh=0.5, purity_thresh=0.7):
    active = get_component_activity(model)
    try:
        return np.concatenate(
         [model.records['credible_sets'][k]
          for k in range(model.dims['K'])
          if active[k]]
        )
    except Exception:
        return np.array([])


def score_finemapping(model, sim, active_thresh=0.5, purity_thresh=0.7):
    credible_sets = model.records['credible_sets']
    causal_snps = model.snp_ids[sim.causal_snps]
    K = model.dims['K']
    active = get_component_activity(model)
    cs_with_causal = np.array(
        [np.any(np.isin(causal_snps, credible_sets[k]))
         for k in range(K) if active[k]]).sum()
    active_components = active.sum()
    num_causal = causal_snps.size
    snps_in_cs = all_credible_snps(model)
    causal_in_cs = np.isin(causal_snps, snps_in_cs).sum()

    row = {
        'score': 'finemapping',
        'active_components': active_components,
        'causal_in_cs': causal_in_cs,
        'cs_with_causal': cs_with_causal,
        'causal_snps': causal_snps.size,
        'snps_in_cs': np.unique(snps_in_cs).size
    }
    return row


p_coloc = lambda model, t1, t2: 1 - \
    np.exp(np.sum(np.log(1e-10 + 1 - model.active[t1] * model.active[t2])))
p_coloc_in_active = lambda model, active, t1, t2: \
    1 - np.exp(np.sum(np.log(
        1e-10 + 1 - model.active[t1, active] * model.active[t2, active])))


def score_coloc(model, sim, thresh=0.99):
    """
    compute true/false positive/negative frequency
    from q(s) for all components
    """
    tril = np.tril_indices(model.dims['T'], k=-1)
    true_coloc = (sim.true_effects @ sim.true_effects.T != 0)[tril]

    model_coloc = np.concatenate(
        [[p_coloc(model, t1, t2) for t1 in range(t2)]
          for t2 in range(model.dims['T'])])
    model_coloc = model_coloc > thresh
    return {
        'score': 'coloc',
        'true_positive': (true_coloc & model_coloc).sum(),
        'false_positive': (~true_coloc & model_coloc).sum(),
        'true_negative': (~true_coloc & ~model_coloc).sum(),
        'false_negative': (true_coloc & ~model_coloc).sum()
    }


def score_active_component_coloc(model, sim, thresh=0.99):
    """
    compute true/false positive/negative frequency
    from q(s) for all ACTIVE components
    ACTIVE = (at least one study p > 0.5), (purity > 0.7)
    """
    tril = np.tril_indices(model.dims['T'], k=-1)
    true_coloc = (sim.true_effects @ sim.true_effects.T != 0)[tril]
    active = get_component_activity(model)
    model_coloc = np.concatenate(
        [[p_coloc_in_active(model, active, t1, t2)
          for t1 in range(t2)] for t2 in range(model.dims['T'])])
    model_coloc = model_coloc > thresh
    return {
        'score': 'coloc_in_active',
        'true_positive': (true_coloc & model_coloc).sum(),
        'false_positive': (~true_coloc & model_coloc).sum(),
        'true_negative': (~true_coloc & ~model_coloc).sum(),
        'false_negative': (true_coloc & ~model_coloc).sum()
    }


def score_causal_component_coloc(model, sim, thresh=0.99):
    """
    compute true/false positive/negative frequency
    from q(s) for all ACTIVE components with a causal SNP
    ACTIVE = (at least one study p > 0.5), (purity > 0.7)
    """
    tril = np.tril_indices(model.dims['T'], k=-1)
    true_coloc = (sim.true_effects @ sim.true_effects.T != 0)[tril]

    # get components that are active and contain a causal snp
    credible_sets = model.records['credible_sets']
    causal_snps = model.snp_ids[sim.causal_snps]
    K = model.dims['K']
    active = get_component_activity(model)
    cs_has_causal = np.array(
        [np.any(np.isin(causal_snps, credible_sets[k])) for k in range(K)])
    active = active & cs_has_causal
    model_coloc = np.concatenate(
        [[p_coloc_in_active(model, active, t1, t2)
        for t1 in range(t2)] for t2 in range(model.dims['T'])])
    model_coloc = model_coloc > thresh
    return {
        'score': 'coloc_in_causal',
        'true_positive': (true_coloc & model_coloc).sum(),
        'false_positive': (~true_coloc & model_coloc).sum(),
        'true_negative': (~true_coloc & ~model_coloc).sum(),
        'false_negative': (true_coloc & ~model_coloc).sum()
    }


def score_matched_component_coloc(model, sim, thresh=0.99):
    """
    compute true/false positive/negative frequency
    from q(s) for all ACTIVE components with a causal SNP
    MATCHED: compute true/false pos/neg frequency for each causal snp s
    ACTIVE = (at least one study p > 0.5), (purity > 0.7)
    """
    tril = np.tril_indices(model.dims['T'], k=-1)

    K = model.dims['K']
    causal_snps = model.snp_ids[sim.causal_snps]
    credible_sets = model.records['credible_sets']
    active = get_component_activity(model)

    component_causal = np.array([np.isin(causal_snps, credible_sets[k]) for k in range(K)]).T
    component_causal = component_causal * active
    active = get_component_activity(model)

    r = {
        'true_positive': 0,
        'true_negative': 0,
        'false_positive': 0,
        'false_negative': 0
    }
    for i, causal_snp in enumerate(causal_snps):
        # studys colocalizing with this causal snp
        true_coloc = (np.outer(sim.true_effects[:, i], sim.true_effects[:, i]) != 0)[tril]
        # components that contain this causal snp
        model_coloc = np.concatenate([[
            p_coloc_in_active(model, component_causal[i], t1, t2)
            for t1 in range(t2)] for t2 in range(model.dims['T'])])
        model_coloc = model_coloc > thresh
        r['true_positive'] += (true_coloc & model_coloc).sum()
        r['false_positive'] += (~true_coloc & model_coloc).sum()
        r['true_negative'] += (~true_coloc & ~model_coloc).sum()
        r['false_negative'] += (true_coloc & ~model_coloc).sum()
    r['score'] = 'coloc_in_matched'
    return r


def score_effect_size_error(model, sim):
    causal_snp_error = model.expected_effects[:, sim.causal_snps] - sim.true_effects
    non_causal_error = np.delete(model.expected_effects, sim.causal_snps, axis=1)
    all_error = np.concatenate([causal_snp_error.flatten(), non_causal_error.flatten()])
    return {
        'score': 'effect_size_error',
        'mean_error': np.mean(all_error),
        'mean_squared_error': np.mean(all_error**2),
        'causal_effect_error': np.mean(causal_snp_error),
        'active_mean_squared_error': np.mean(causal_snp_error**2),
        'null_mean_error': np.mean(non_causal_error),
        'null_mean_squared_error': np.mean(non_causal_error**2)
    }


score_functions = {
    'finemapping': score_finemapping,
    'coloc': score_coloc,
    'coloc_in_active': score_active_component_coloc,
    'coloc_in_causal': score_causal_component_coloc,
    'coloc_matched': score_matched_component_coloc,
    'effect_size_error': score_effect_size_error
}

def score(row):
    """
    score a simulation using score_functions
    return a list of dictionaries, one for each row
    can turn this into a dataframe with pd.DataFrame(score(row))
    """
    try:
        sim = pickle.load(open(row[1].sim_path, 'rb'))
        model = load(row[1].model_path)
        scores = [f(model, sim) for f in score_functions.values()]
        row_dict = row[1].to_dict()
        [s.update(row_dict) for s in scores]
        return scores
    except Exception:
        return []


make_model_path = lambda x: '{}{}'.format(x.sim_path[:-14], x.model_key)

def gen_sim_table(sim_spec, model_spec):
    """
    combine sim
    """
    a = sim_spec.sim_id
    b = model_spec.model_key
    index = pd.MultiIndex.from_product([a, b], names = ['sim_id', 'model_key'])
    sim_table = pd.DataFrame(index = index).reset_index()
    sim_table = sim_table.merge(sim_spec, on='sim_id').merge(model_spec, on='model_key')
    sim_table.loc[:, 'model_path'] = sim_table.apply(make_model_path, axis=1)
    return sim_table

