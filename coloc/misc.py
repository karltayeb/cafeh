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
    expression_path = '/work-zfs/abattle4/karl/cosie_analysis/output/GTEx/chr16/{}/{}.expression'.format(gene, gene)
    genotype_path = '/work-zfs/abattle4/karl/cosie_analysis/output/GTEx/chr16/{}/{}.raw'.format(gene, gene)

    model = pickle.load(open(model_path, 'rb'))

    model.weight_means = np.zeros((model.dims['T'],model.dims['K'],model.dims['N']))
    model.weight_vars = np.ones((model.dims['T'],model.dims['K'],model.dims['N']))

    model.weight_means[:, :, model.records['snp_subset']] = model.records['mini_wm']
    model.weight_vars[:, :, model.records['snp_subset']] = model.records['mini_wv']

    if load_data:
        data = make_gtex_genotype_data_dict(expression_path, genotype_path)
        model.__dict__.update(data)
    return model


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
        color_snps = np.isin(model.snp_ids, model.record_credible_sets[0][k])
        ax[1].scatter(pos[color_snps], pip[color_snps], s=100, marker='o', alpha=0.5, label='component {}'.format(k))

    ax[1].set_xlabel('position')
    ax[0].set_xlabel('component')
    ax[0].set_ylabel('PIP')
    ax[0].set_title('expected effect size')
    ax[1].set_title('PIP')
    plt.legend()
