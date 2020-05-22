import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from .kls import categorical_kl

def need_to_flip(variant_id):
    _, _, major, minor, _, ref = variant_id.strip().split('_')
    if minor != ref:
        return True
    else:
        return False


flip = lambda x: (x-1)*-1 + 1


def load_gtex_genotype(genotype_path, standardize=False):
    """
    fetch genotype
    flip codings that are need to be flipped
    set snp_ids to be consistent with gtex
    """
    #load genotype
    genotype = pd.read_csv(genotype_path, sep=' ')
    genotype = genotype.set_index('IID').iloc[:, 5:]

    # recode genotypes
    coded_snp_ids = np.array([x.strip() for x in genotype.columns])
    snp_ids = np.array(['_'.join(x.strip().split('_')[:-1]) for x in coded_snp_ids])
    snp_ids = {x: '_'.join(x.strip().split('_')[:-1]) for x in coded_snp_ids}
    genotype.rename(columns=snp_ids, inplace=True)

    flips = np.array([need_to_flip(vid) for vid in coded_snp_ids])
    genotype.iloc[:, flips] = genotype.iloc[:, flips].applymap(flip) 
    return genotype


def load_gtex_expression(expression_path):
    """
    load expression, drop unexpressed individuals
    """
    # load expression
    gene_expression = pd.read_csv(expression_path, sep='\t', index_col=0)
    # drop individuals that do not have recorded expression
    gene_expression = gene_expression.loc[:, ~np.all(np.isnan(gene_expression), 0)]
    return gene_expression


def make_gtex_genotype_data_dict(expression_path, genotype_path, standardize=False):
    # load expression
    gene_expression = load_gtex_expression(expression_path)
    genotype = load_gtex_genotype(genotype_path)

    # center, mean immpute
    genotype = (genotype - genotype.mean(0))
    genotype = genotype.fillna(0)

    # standardize
    if standardize:
        genotype = genotype / genotype.std(0)

    # filter down to common individuals
    individuals = np.intersect1d(genotype.index.values, gene_expression.columns.values)
    genotype = genotype.loc[individuals]
    gene_expression = gene_expression.loc[:, individuals]

    # load covariates
    covariates = pd.read_csv('/work-zfs/abattle4/karl/cosie_analysis/output/GTEx/covariates.csv', sep='\t', index_col=[0, 1])
    covariates = covariates.loc[gene_expression.index]
    covariates = covariates.loc[:, genotype.index.values]
    X = genotype.values.T
    data = {
        'X': X,
        'Y': gene_expression.values,
        'covariates': covariates,
        'snp_ids': genotype.columns.values,
        'sample_ids': genotype.index.values,
        'tissue_ids': gene_expression.index.values
    }
    return data


def compute_summary_stats(data):
    B = {}
    V = {}
    S = {}
    for i, tissue in enumerate(data['tissue_ids']):
        cov = data['covariates'].loc[tissue]
        mask = ~np.isnan(cov.iloc[0])
        cov = cov.values[:, mask]
        y = data['Y'][i, mask]
        X = data['X'][:, mask]

        #H = cov.T @ np.linalg.solve(cov @ cov.T, cov)
        H = (np.linalg.pinv(cov) @ cov)
        yc = y - y @ H
        Xc = X - X @ H
        # prep css data
        B[tissue] = (Xc @ yc) / np.einsum('ij,ij->i', Xc, Xc)
        r = yc - B[tissue][:, None]*Xc
        V[tissue] = np.einsum('ij,ij->i', r, r) / np.einsum('ij,ij->i', Xc, Xc) / (yc.size)
        S[tissue] = np.sqrt(B[tissue]**2/yc.size + V[tissue])

    B = pd.DataFrame(B, index=data['snp_ids']).T
    V = pd.DataFrame(V, index=data['snp_ids']).T
    S = pd.DataFrame(S, index=data['snp_ids']).T
    return B, S, V


def get_gtex_summary_stats(ap):
    associations = pd.read_csv(ap)
    associations.loc[:, 'sample_size'] = (associations.ma_count / associations.maf / 2)
    Ba = associations.pivot('tissue', 'variant_id', 'slope')
    Va = associations.pivot('tissue', 'variant_id', 'slope_se')**2
    n = associations.pivot('tissue', 'variant_id', 'sample_size')
    Sa = np.sqrt(Ba**2/n + Va)
    return Ba, Sa, Va, n


def rehydrate_model(model):
    model.weight_means = np.zeros((model.dims['T'],model.dims['K'],model.dims['N']))
    model.weight_vars = np.ones((model.dims['T'],model.dims['K'],model.dims['N']))

    model.weight_means[:, :, model.records['snp_subset']] = model.records['mini_wm']
    model.weight_vars[:, :, model.records['snp_subset']] = model.records['mini_wv']


def load_model(model_path, load_data=False):
    gene = model_path.split('/')[-2]
    base_path = '/'.join(model_path.split('/')[:-1])
    expression_path = '{}/{}.expression'.format(base_path, gene)
    genotype_path = '{}/{}.raw'.format(base_path, gene)

    model = pickle.load(open(model_path, 'rb'))

    model.weight_means = np.zeros((model.dims['T'],model.dims['K'],model.dims['N']))
    model.weight_vars = np.ones((model.dims['T'],model.dims['K'],model.dims['N']))

    model.weight_means[:, :, model.records['snp_subset']] = model.records['mini_wm']
    model.weight_vars[:, :, model.records['snp_subset']] = model.records['mini_wv']

    if load_data:
        data = make_gtex_genotype_data_dict(expression_path, genotype_path)
        model.__dict__.update(data)
    return model


def compute_records(model):
    """
    save the model with data a weight parameters removed
    add 'mini_weight_measn' and 'mini_weight_vars' to model
    the model can be reconstituted in a small number of iterations
    """
    PIP = 1 - np.exp(np.log(1 - model.pi + 1e-10).sum(0))
    mask = (PIP > 0.01)
    wv = model.weight_vars[:, :, mask]
    wm = model.weight_means[:, :, mask]

    credible_sets, purity = model.get_credible_sets(0.99)
    active = np.array([purity[k] > 0.5 for k in range(model.dims['K'])])
    records = {
        'active': active,
        'purity': purity,
        'credible_sets': credible_sets,
        'EXz': model.pi @ model.X,
        'mini_wm': wm,
        'mini_wv': wv,
        'snp_subset': mask
    }
    model.records = records


def strip_and_dump(model, path, save_data=False):
    """
    save the model with data a weight parameters removed
    add 'mini_weight_measn' and 'mini_weight_vars' to model
    the model can be reconstituted in a small number of iterations
    """
    compute_records(model)

    # purge precompute
    for key in model.precompute:
        model.precompute[key] = {}
    model.__dict__.pop('weight_means', None)
    model.__dict__.pop('weight_vars', None)
    if not save_data:
        model.__dict__.pop('X', None)
        model.__dict__.pop('Y', None)
        model.__dict__.pop('covariates', None)
    pickle.dump(model, open(path, 'wb'))


def repair_model(model_path):
    gene = model_path.split('/')[-2]
    base_path = '/'.join(model_path.split('/')[:-1])
    expression_path = '{}/{}.expression'.format(base_path, gene)
    genotype_path = '{}/{}.raw'.format(base_path, gene)

    X = make_gtex_genotype_data_dict(expression_path, genotype_path)['X']

    model = pickle.load(open(model_path, 'rb'))
    compute_records(model)
    strip_and_dump(model, model_path)


def compute_pip(model):
    active = model.records['active']
    return 1 - np.exp(np.log(1 - model.pi + 1e-10).sum(0))


def component_scores(model):
    purity = model.get_credible_sets(0.99)[1]
    active = np.array([purity[k] > 0.5 for k in range(model.dims['K'])])
    if active.sum() > 0:
        mw = model.weight_means
        mv = model.weight_vars
        pi = model.pi
        scores = np.einsum('ijk,jk->ij', mw / np.sqrt(mv), model.pi)
        weights = pd.DataFrame(
            scores[:, active],
            index = model.tissue_ids,
            columns = np.arange(model.dims['K'])[active]
        )
    else:
        weights = pd.DataFrame(
            np.zeros((model.dims['T'], 1)),
            index = model.tissue_ids
        )
    return weights


def kl_components(m1, m2, a1=None, a2=None, purity_threshold=0.0):
    """
    pairwise kl of components for 2 models
    """
    if a1 is None:
        a1 = np.ones(m1.dims['K']) > 0
    if a2 is None:
        a2 = np.ones(m2.dims['K']) > 0
    #a1 = m1.records['active']
    #a2 = m2.records['active']

    kls = np.array([[
        categorical_kl(m1.pi[k1], m2.pi[k2])
        + categorical_kl(m2.pi[k2], m1.pi[k1])
        for k1 in range(m1.dims['K']) if a1[k1]]
        for k2 in range(m2.dims['K']) if a2[k2]])
    return kls


def make_variant_report(model, gene):
    PIP = 1 - np.exp(np.log(1 - model.pi + 1e-10).sum(0))
    purity = model.get_credible_sets(0.99)[1]
    active = np.array([purity[k] > 0.5 for k in range(model.dims['K'])])
    if active.sum() == 0:
        active[0] = True

    pi = pd.DataFrame(model.pi.T, index=model.snp_ids)
    cset_alpha = pd.concat(
        [pi.iloc[:, k].sort_values(ascending=False).cumsum() - pi.iloc[:, k]
         for k in np.arange(model.dims['K']) if active[k]],
        sort=False, axis=1
    )

    most_likely_k = np.argmax(pi.values[:, active], axis=1)
    most_likely_p = np.max(pi.values[:, active], axis=1)
    most_likely_cset_alpha = cset_alpha.values[np.arange(pi.shape[0]), most_likely_k]

    A = pd.DataFrame(
        [PIP, most_likely_k, most_likely_p, most_likely_cset_alpha],
        index=['PIP','k', 'p', 'min_alpha'], columns=model.snp_ids).T

    A.loc[:, 'chr'] = [x.split('_')[0] for x in A.index]
    A.loc[:, 'start'] = [int(x.split('_')[1]) for x in A.index]
    A.loc[:, 'end'] = A.loc[:, 'start'] + 1
    A.reset_index(inplace=True)
    A.loc[:, 'variant_id'] = A.loc[:, 'index'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    A.loc[:, 'ref'] = A.loc[:, 'index'].apply(lambda x: x.split('_')[-1])


    A.loc[:, 'gene_id'] = gene
    A = A.set_index(['chr', 'start', 'end'])
    return A.loc[:, ['gene_id', 'variant_id', 'ref', 'PIP', 'k', 'p', 'min_alpha']]
