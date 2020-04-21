import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def make_gtex_genotype_data_dict(expression_path, genotype_path):
    # load expression
    gene_expression = pd.read_csv(expression_path, sep='\t', index_col=0)
    #load genotype
    genotype = pd.read_csv(genotype_path, sep=' ')
    genotype = genotype.set_index('IID').iloc[:, 5:]
    # center, mean immpute
    genotype = (genotype - genotype.mean(0))
    genotype = genotype.fillna(0)
    # standardize
    genotype = genotype / genotype.std(0)
    # drop individuals that do not have recorded expression
    gene_expression = gene_expression.loc[:, ~np.all(np.isnan(gene_expression), 0)]
    # filter down to relevant individuals
    genotype = genotype.loc[gene_expression.columns]
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


def plot_components(model, figsize=(15,5)):
    active = model.records['active']
    pip = compute_pip(model)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(
        model.get_expected_weights()[:, active], yticklabels=model.tissue_ids,
        cmap='RdBu', center=0, ax=ax[0])

    pos = np.array([int(x.split('_')[1]) for x in model.snp_ids])
    ax[1].scatter(pos, pip, c='k', marker='x')

    for k in np.arange(model.dims['K'])[active]:
        color_snps = np.isin(model.snp_ids, model.records['credible_sets'][k])
        ax[1].scatter(pos[color_snps], pip[color_snps], s=100, marker='o', alpha=0.5, label='component {}'.format(k))

    ax[1].set_xlabel('position')
    ax[0].set_xlabel('component')
    ax[0].set_ylabel('PIP')
    ax[0].set_title('expected effect size')
    ax[1].set_title('PIP')
    plt.legend()


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

def kl_components(m1, m2, purity_threshold=0.0):
    """
    pairwise kl of components for 2 models
    """
    a1 = m1.records['active']
    a2 = m2.records['active']
    kls = np.array([[
        categorical_kl(m1.pi[k1], m2.pi[k2])
        + categorical_kl(m2.pi[k2], m1.pi[k1])
        for k1 in range(20) if a1[k1]]
        for k2 in range(20) if a2[k2]])
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
         for k in np.arange(model.dims['K']) if purity[k] > 0.5],
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
