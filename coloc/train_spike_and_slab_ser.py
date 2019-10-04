import numpy as np
import pandas as pd
from itertools import product
import pickle
import sys
from spike_and_slab_ser import update_ss_weights, update_pi

def get_inputs(zscore_path, ld_path, gene):
    X = pd.read_csv(ld_path + gene, index_col=0)
    zscores = pd.read_csv(zscore_path + gene + '.zscore_matrix.txt', '\t', index_col=0)

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
zscores_path = '/work-zfs/abattle4/marios/GTEx_v8/coloc/zscore_genes_for_Karl/'
ld_path = '/work-zfs/abattle4/karl/marios_correlation_matrices/'
output_dir = '/work-zfs/abattle4/karl/ss_ser/models/'

# idx = int(sys.argv[1])

genes = [
'ENSG00000073464.11',
'ENSG00000160181.8',
'ENSG00000100258.17',
'ENSG00000164904.17',
'ENSG00000135362.13',
'ENSG00000178172.6',
'ENSG00000141644.17',
'ENSG00000184293.7',
'ENSG00000141934.9',
'ENSG00000185238.12'
]

Ks = [5, 10, 20]
prior_variances = [0.1, 1.0]
postfixes = np.arange(10)

states = list(product(postfixes, prior_variances, Ks, genes))
for postfix, prior_variance, K, gene in states:

    print('Training mixed ser for gene {}:\n\tK={}\nSaving outputs to {}'.format(gene, K, output_dir))
    save_path = '{}/gene{}_sigma2{}_K{}_{}_model'.format(output_dir, gene, prior_variance, K, postfix)

    ###################
    # get Y and X #
    ###################
    X, Y, tissues, snp_ids = get_inputs(zscores_path, ld_path, gene)
    T, N = Y.shape


    #######################
    # set hyperparameters #
    #######################
    prior_activity = np.exp(-1 * np.linspace(1, 10, K))

    #######################################
    # initialize (variational) parameters #
    #######################################
    pi = np.random.random((N, K)) + 1
    pi = pi / pi.sum(0)

    weights = np.random.random((T, K)) * 5
    active = np.ones((T, K))

    # make save paths
    convergence_status = False

    for i in range(maxiter):
        # update pi and beta
        for j in range(100):
            diff = update_pi(X, Y, weights, active, pi)

            # exit inner loop if we converged
            if diff < 1e-8:
                break

        # update weights/active probabilities
        diff = update_ss_weights(X, Y, weights, active, pi, prior_activity, prior_variance)

        if i % 100 == 0:
            print('iter {} outer loop elbo: {}, max_weight_diff: {}'.format(
                i, (0), diff))

        if diff < 1e-8:
            print('parameters converged at iter {}: diff={}'.format(i, diff))
            convergence_status = True
            break

    save_dict = {
        'pi': pi,
        'active': active,
        'weights': weights,
        'prior_activity': prior_activity,
        'prior_variance': prior_variance,
        'K': K,
        'N': Y.shape[1],
        'T': Y.shape[0],
        'gene': gene,
        'converged': convergence_status,
        'snp_ids': snp_ids,
        'tissues': tissues
    }
    pickle.dump(save_dict, open(save_path, 'wb'))
