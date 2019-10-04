import sys
import numpy as np
import pandas as pd
import pickle
from collections import namedtuple
from mixed_ser import *
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product


def get_inputs(zscore_path, ld_path, gene):
    X = pd.read_csv(ld_path + gene, index_col=0)
    zscores = pd.read_csv(zscore_path + gene + '.zscore_matrix.txt', '\t', index_col=0)

    nan_snps = np.all(np.isnan(X.values), axis=1)
    X = X.iloc[~nan_snps].iloc[:, ~nan_snps]

    active_snps = np.isin(X.index, zscores.index)
    X = X.iloc[active_snps].iloc[:, active_snps]

    active_snps = np.isin(zscores.index, X.index)
    Y = zscores.iloc[active_snps]

    tissues = Y.columns.values
    snp_ids = Y.index.values
    pos = np.array([int(snp_id.split('_')[1]) for snp_id in snp_ids])

    Y = Y.T.values
    X = X.values
    X = (X + np.eye(X.shape[0])*1e-6) / (1+1e-6)
    
    return X, Y, tissues, snp_ids

def restart_components(pi, beta_means, beta_vars, weights):
    """
    reinitialize components that have died in optimization
    """
    not_active = np.abs(weights).max(0) < 1e-6
    if not_active.sum() > 0:
        print('restarting componenents {}'.format(not_active))

        restart_freq = (1 / (pi.sum(1) + 0.01))
        restart_freq = restart_freq / restart_freq.sum()
        for c in np.arange(K)[not_active]:
            pi[:, c] = restart_freq
            weights[:, c] = (np.random.random(T) + 1)
            beta_means[c] = 1.0
            beta_vars[c] = 1.0

def update_variational_params(X, Y, pi, beta_means, beta_vars, weights):
    """
    on iteration through pi/beta updates
    returns max difference in parameters to monitor convergence
    """
    pi_old = pi.copy()
    for k in range(pi.shape[1]):
        # if component is being used, update
        if np.abs(weights[:, k]).max() > 1e-5:
            update_pi(X, Y, pi, beta_means, beta_vars, weights, k)
            update_beta(X, Y, pi, beta_means, beta_vars, weights, k)

    pi_diff = np.abs(pi - pi_old).max()
    return pi_diff

def update_weights(X, Y, pi, beta_means, beta_vars, weights, penalty, problem, param_dict):
    """
    set weights in cvxpy problem
    returns max absolute difference between old weights and new weights
    """
    set_params(pi, beta_means, beta_vars, penalty, param_dict)
    old_weights = weights.copy()
    weights[:, :] = np.array([solve_w_tissue(Yt, param_dict, problem) for Yt in Y])
    weight_diff = np.abs(weights - old_weights).max()
    return weight_diff


if __name__ == "__main__":

    maxiter = 1000
    max_restart = 10

    #####################
    # GET SCRIPT INPUTS #
    #####################


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
    output_dir = '/work-zfs/abattle4/karl/mixed_ser/models/'

    idx = int(sys.argv[1])

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

    Ks = [10, 20]
    penalties = [0.1, 1, 10, 20]
    nonnegs = [True]
    postfixes = np.arange(5)

    states = list(product(genes, Ks, penalties, nonnegs, postfixes))
    gene, K, penalty, nonneg, postfix = states[idx]
    """
    print('Training mixed ser for gene {}:\n\tK={}\n\tpenalty={}\n\tnonneg={}\nSaving outputs to {}'.format(gene, K, penalty, nonneg, output_dir))
    ###################
    # get Y and X #
    ###################

    X, Y, tissues, snp_ids = get_inputs(zscores_path, ld_path, gene)
    T, N = Y.shape

    #######################################
    # initialize (variational) parameters #
    #######################################
    pi = np.random.random((N, K)) + 1
    pi = pi / pi.sum(0)

    beta_means = np.random.random(K)
    beta_vars = np.ones(K)
    weights = np.random.random((T, K)) + 1

    problem, param_dict = make_problem(N, K, True)

    #########################
    # make mvn distribution #
    #########################

    dist = multivariate_normal(cov=X)
    chol = np.linalg.cholesky(X)


    # make save paths
    model_save_path = '{}/gene{}_K{}_lambda{}_nonneg{}_{}_model'.format(output_dir, gene, K, penalty, nonneg, postfix)
    fig_save_path = '{}/gene{}_K{}_lambda{}_nonneg{}_{}_plot.png'.format(output_dir, gene, K, penalty, nonneg, postfix)

    elbos = [compute_elbo(X, Y, pi, beta_means, beta_vars, weights, dist, penalty=penalty)]
    convergence_status = False

    for i in range(maxiter):
        # restart dead components if its early in optimization
        if i < max_restart:
            restart_components(pi, beta_means, beta_vars, weights)

        # update pi and beta
        for j in range(100):
            pi_diff = update_variational_params(X, Y, pi, beta_means, beta_vars, weights)

            # exit inner loop if we converged
            if pi_diff < 1e-8:
                # enter a new outer loop, need this?
                convergence_status = False
                break

        # update weights
        weight_diff = update_weights(X, Y, pi, beta_means, beta_vars, weights, penalty, problem, param_dict)
        elbos.append(compute_elbo(X, Y, pi, beta_means, beta_vars, weights, dist, penalty=penalty))

        if i % 10 == 0:
            print('iter {} outer loop elbo: {}, max_weight_diff: {}'.format(
                i, (elbos[-1] - elbos[-2]), weight_diff))
        if weight_diff < 1e-8:
            print('weight parameter converged')
            convergence_status = True
            break


        if i % 50 == 0:
            # save model
            save_dict = {
                'pi': pi,
                'weights': weights,
                'beta_means': beta_means,
                'beta_vars': beta_vars,
                'elbos': elbos,
                'N': N,
                'T': T,
                'K': K,
                'gene': gene,
                'converged': convergence_status,
                'snp_ids': snp_ids,
                'tissues': tissues}

            pickle.dump(save_dict, open(model_save_path, 'wb'))  

    """
    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for k in range(K):
        ax[0].scatter(np.arange(N)[pi[:, k] > 2/N], pi[:, k][pi[:, k] > 2/N], alpha=0.5)

    active = np.abs(weights).max(0) > 1e-6
    sns.heatmap(weights[:, active], annot=True, cmap='RdBu_r', ax=ax[1])
    fig.savefig(fig_save_path)
    plt.close()
    """

    # save model
    save_dict = {
        'pi': pi,
        'weights': weights,
        'beta_means': beta_means,
        'beta_vars': beta_vars,
        'elbos': elbos,
        'N': N,
        'T': T,
        'K': K,
        'gene': gene,
        'converged': convergence_status,
        'snp_ids': snp_ids,
        'tissues': tissues}

    pickle.dump(save_dict, open(model_save_path, 'wb'))  
