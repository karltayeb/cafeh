import numpy as np
from scipy.stats import norm
import pandas as pd
from .kls import unit_normal_kl, normal_kl, categorical_kl

#########################
# MODEL QUERY FUNCTIONS #
#########################
def check_convergence(self):
    if len(self.elbos) > 2:
        delta = np.abs((self.elbos[-1] - self.elbos[-2]) / self.elbos[-2])
        if delta < self.tolerance:
            return True
    else:
        return False

def get_expected_weights(self):
    if self.weight_means.ndim == 2:
        weights = self.weight_means * self.active
    else:
        weights = np.einsum('ijk,kj->ij', self.weight_means * self.active[:, :, None], self.pi.T)
    return weights

def get_top_snp_per_component(self):
    """
    returns snp with highest posterior probability for each component
    returns p(max_snp) for each components
    """
    return self.snp_ids[self.pi.T.argmax(axis=0)], self.pi.T.max(axis=0)

def _get_cs(pi, alpha):
    argsort = np.flip(np.argsort(pi))
    cset_size = (np.cumsum(pi[argsort]) < alpha).sum() + 1
    return argsort[:cset_size]

def get_credible_sets(self, alpha=0.95):
    """
    return {k: credible_set}
    """
    credible_sets = {k: _get_cs(
        self.pi[k], alpha) for k in range(self.dims['K'])}
    return credible_sets

def get_purity(self, alpha=0.95):
    """
    compute purity of credible credible sets
    if the cs contains > 100 SNPs compute from random sample of 100 SNPs
    """
    credible_sets = {k: _get_cs(self.pi[k], alpha) for k in range(self.dims['K'])}
    purity = {}
    for k in range(self.dims['K']):
        if credible_sets[k].size == 1:
            purity[k] = 1.0
        elif credible_sets[k].size < 100:
            LD = self.get_ld(credible_sets[k])
            purity[k] = np.min(np.abs(LD))
        else:
            cs = np.random.choice(credible_sets[k], 100)
            LD = self.get_ld(cs)
            purity[k] = np.min(np.abs(LD))

    return purity

def get_study_pip(self):
    pip = []
    for t in range(self.dims['T']):
        pip.append(1 - np.exp(np.sum(np.log((1 - self.pi * self.active[t][:, None] + 1e-100)), 0)))
    return pd.DataFrame(np.array(pip), columns=self.snp_ids, index=self.study_ids)

def get_pip(self):
    # probability that each component is active in atleast one tissue
    active = 1 - np.exp(np.sum(np.log(1 - self.active + 1e-100), 0))
    # compute pip
    pip = 1 - np.exp(np.sum(np.log((1 - self.pi * active[:, None] + 1e-100)), 0))
    return pip


def _get_minalpha(pi):
    """
    report the minimum alpha value to include this snp in cs
    """
    argsort = np.flip(np.argsort(pi))
    resort = np.argsort(argsort)
    cumsum = np.cumsum(pi[argsort])
    minalpha = np.roll(cumsum, 1)
    minalpha[0] = 0
    return minalpha[resort]

def get_minalpha(model):
    return  pd.DataFrame(
        np.array([_get_minalpha(model.pi[k]) for k in range(model.dims['K'])]).T,
        index=model.snp_ids
    )

def summary_table(model, filter_variants=True):
    """
    map each variant to its top component
    report alpha, rank, pi, effect, effect_var in top component
    report p_active (component level stat) and pip (variant level stat)
    """

    study_pip = model.get_study_pip().T
    table = study_pip.reset_index().melt(id_vars='index').rename(columns={
        'index': 'variant_id',
        'variable': 'study',
        'value': 'pip' 
    })

    top_component = pd.Series(model.pi.argmax(0), index=model.snp_ids).to_dict()
    table.loc[:, 'top_component'] = table.variant_id.apply(lambda x: top_component.get(x))

    minalpha = get_minalpha(model).to_dict()
    table.loc[:, 'alpha'] = [minalpha.get(k).get(v) for k, v in zip(table.top_component.values, table.variant_id.values)]

    rank = pd.DataFrame({k: np.argsort(np.flip(np.argsort(model.pi[k]))) for k in range(model.dims['K'])}, index=model.snp_ids).to_dict()
    table.loc[:, 'rank'] = [rank.get(k).get(v) for k, v in zip(table.top_component.values, table.variant_id.values)]

    active = pd.DataFrame(model.active, index=model.study_ids)
    active.loc['all'] = (model.active.max(0) > 0.5).astype(int)
    active = active.to_dict()
    table.loc[:, 'p_active'] = [active.get(k).get(s) for k, s in zip(table.top_component.values, table.study.values)]

    pi = pd.Series(model.pi.max(0), index=model.snp_ids).to_dict()
    table.loc[:, 'pi'] = table.variant_id.apply(lambda x: pi.get(x))

    if filter_variants:
        rank = model.pi.max(0).argsort().argsort()
        min_snps = model.snp_ids[rank[-100:]]
        table = table[(table.p_active > 0.5) | (table.variant_id.isin(min_snps))] #.sort_values(by=['chr', 'start'])

    # add effect size and variance
    study2idx = {s: i for i, s in enumerate(model.study_ids)}
    var2idx = {s: i for i, s in enumerate(model.snp_ids)}
    table.loc[:, 'effect'] = [model.weight_means[study2idx.get(s), c, var2idx.get(v)] for s, v, c in zip(
        table.study, table.variant_id, table.top_component)]
    table.loc[:, 'effect_var'] = [model.weight_vars[study2idx.get(s), c, var2idx.get(v)] for s, v, c in zip(
        table.study, table.variant_id, table.top_component)]
    return table


def coloc_table(model, phenotype, **kwargs):
    """
    generate a table giving colocalization with respect to a particular phenotype
    model: cafeh instance
    phenotype: phenotype_id we want to generte coloc table from
    """
    p_active = pd.DataFrame(model.active, index=model.study_ids)
    p_active = p_active.reset_index().melt(id_vars='index')

    p_coloc = pd.DataFrame((model.active * model.active[model.study_ids==phenotype]), index=model.study_ids)
    p_coloc = p_coloc.reset_index().melt(id_vars='index')

    p_active.loc[:, 'p_coloc'] = p_coloc.value
    p_active = p_active.rename(columns={'index': 'tissue', 'variable': 'component', 'value': 'p_active'})

    p_active.loc[:, 'phenotype'] = phenotype
    for key, val in kwargs.items():
        p_active.loc[:, key] = val
    return p_active


def get_component_coloc(self):
    """
    returns K x T x T matrix for probability of colocalization of two studys in a component
    """
    weight_var = 1 / (1 + (1 /self.prior_variance))
    sign_error = norm.cdf(-np.abs(self.weights) / np.sqrt(weight_var))

    coloc = np.zeros((self.dims['K'], self.dims['T'], self.dims['T']))
    for k in range(self.dims['K']):
        right_signs = np.outer(1 - sign_error[:, k], 1 - sign_error[:, k])
        right_signs[np.arange(self.dims['T']), np.arange(self.dims['T'])] = 1 - sign_error[:, k]
        
        both_active = np.outer(self.active[:, k], self.active[:, k])
        both_active[np.arange(self.dims['T']), np.arange(self.dims['T'])] = self.active[:, k]

        coloc[k] = both_active * right_signs
    return coloc

