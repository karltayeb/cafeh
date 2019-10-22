import numpy as np
import pandas as pd
from itertools import product
import pickle
import sys
from spike_and_slab_ser import SpikeSlabSER

def get_inputs(zscore_path, ld_path, afreq_path, gene):
    """
    zscore path: path to directory with zcore files [gene].zscore_matrix.txt'
    ld_path: path to directory with emprical correlation matrices of snps [gene]
    afreq_path: path to directory with allele frequency files, organized by chromasome
    """
    if zscore_path[-1] == '/':
        zscore_path = zscore_path[:-1]
    if ld_path[-1] == '/':
        ld_path = ld_path[:-1]
    if afreq_path[-1] == '/':
        afreq_path = afreq_path[:-1]

    X = pd.read_csv(ld_path, index_col=0)
    zscores = pd.read_csv(zscore_path, '\t', index_col=0)

    nan_snps = np.all(np.isnan(X.values), axis=1)
    X = X.iloc[~nan_snps].iloc[:, ~nan_snps]

    active_snps = np.isin(X.index, zscores.index)
    X = X.iloc[active_snps].iloc[:, active_snps]

    active_snps = np.isin(zscores.index, X.index)
    Y = zscores.iloc[active_snps]
    Y = Y.iloc[:, ~np.any(np.isnan(Y.values), 0)]

    tissues = Y.columns.values
    snp_ids = Y.index.values
    pos = np.array([int(snp_id.split('_')[1]) for snp_id in snp_ids])

    Y = Y.T.values
    X = X.values
    X = (X + np.eye(X.shape[0])*1e-6) / (1+1e-6)

    # flip sign of zscore if alternate allele is major
    chrom = snp_ids[0].split('_')[0][3:]
    afreq = pd.read_csv('{}/chrom{}.afreq'.format(afreq_path, chrom))

    sign = np.ones(snp_ids.size)
    sign[afreq.set_index('ID').loc[snp_ids].ALT_FREQS > 0.5] = -1
    Y = Y * sign
    X = X * np.outer(sign, sign)

    return X, Y, tissues, snp_ids

maxiter = 1000

#####################
# GET SCRIPT INPUTS #
#####################

"""
zscores_path = sys.argv[1]
ld_path = sys.argv[2]
gene = sys.argv[3]
output_dir = sys.argv[4]

postfix = sys.argv[5]
K = int(sys.argv[6])
penalty = float(sys.argv[7])
nonneg = bool(sys.argv[8])

"""

gene = 'ENSG00000239887.4'

zscore_path = '/work-zfs/abattle4/marios/GTEx_v8/coloc/zscore_matrices/rs12143842_ENSG00000239887.4.zscore_matrix.txt'
ld_path = '/work-zfs/abattle4/marios/GTEx_v8/coloc/ld_matrices/rs12143842_ENSG00000239887.4'
afreq_path = '/work-zfs/abattle4/karl/afreq/'
output_dir = '/work-zfs/abattle4/karl/ss_ser/models/'

K = 20
prior_variance = 20.0
prior_activity = np.exp(-1.0)
postfix = 0

model_name = 'gene-{}_sigma2-{:.2f}_phi-{:.2f}_K-{}_run-{}_model'.format(gene, prior_variance, prior_activity, K, postfix)

print('Training spike and slab ser for gene {}:\n\tmodel_name={}\n\tSaving outputs to {}'.format(
    gene, model_name, output_dir))
###################
# get Y and X #
###################
X = pd.read_csv(ld_path, index_col=0)
zscores = pd.read_csv(zscore_path, sep='\t', index_col=0)

nan_snps = np.all(np.isnan(X.values), axis=1)
X = X.iloc[~nan_snps].iloc[:, ~nan_snps]

active_snps = np.isin(X.index, zscores.index)
X = X.iloc[active_snps].iloc[:, active_snps]

active_snps = np.isin(zscores.index, X.index)
Y = zscores.iloc[active_snps]
Y = Y.iloc[:, ~np.any(np.isnan(Y.values), 0)]

tissues = Y.columns.values
snp_ids = Y.index.values
pos = np.array([int(snp_id.split('_')[1]) for snp_id in snp_ids])

Y = Y.T.values
X = X.values
X = (X + np.eye(X.shape[0])*1e-6) / (1+1e-6)

# flip sign of zscore if alternate allele is major
chrom = snp_ids[0].split('_')[0][3:]
afreq = pd.read_csv('{}/chrom{}.afreq'.format(afreq_path, chrom))

sign = np.ones(snp_ids.size)
sign[afreq.set_index('ID').loc[snp_ids].ALT_FREQS > 0.5] = -1
Y = Y * sign
X = X * np.outer(sign, sign)

T, N = Y.shape

###############
#  make model #
###############
model = SpikeSlabSER(
    X=X, Y=Y, K=K,
    snp_ids=snp_ids, tissue_ids=tissues,
    prior_activity=prior_activity * np.ones(K),
    prior_variance=prior_variance
)
model.forward_fit(early_stop=True, verbose=True, restarts=1)
model.save(output_dir, model_name)
