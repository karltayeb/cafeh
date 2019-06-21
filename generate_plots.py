import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys


import gpflow
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

import pickle

def function_means(model, X, full_cov=False):
    """
    Predict latent functions at a set of test points
    assume we are using whitened representation or inducing points
    """
    Kgg = np.array([np.diag(Kggi) for Kggi in model.kern.compute_Kgg(X, X)])
    Z = model.feature.feat.Z.value
    Kgu = model.kern.compute_Kgg(X, Z)
    Kuu = model.kern.compute_Kgg(Z, Z)
    L = np.linalg.cholesky(Kuu + np.eye(Z.shape[0]) *1e-6)
    Li = np.array([np.linalg.inv(l).T for l in L])
    
    q_mu = model.q_mu.value.T
    q_sqrt = model.q_sqrt.value
    
    KguLi = np.einsum('ijk,ikl->ijl', Kgu, Li)
    mu = np.einsum('ijk,ik->ij', KguLi, q_mu)
    
    q_sqrt = model.q_sqrt.value
    KguLiQsqrt = np.einsum('ijk,ikl->ijl', KguLi, q_sqrt)

    if full_cov == True:
        var = Kgg - np.einsum('ijk,ilk->ijl', KguLi, KguLi) + np.einsum('ijk,ilk->ijl', KguLiQsqrt, KguLiQsqrt)
    else:
        var = Kgg - np.einsum('ijk,ijk->ij', KguLi, KguLi) + np.einsum('ijk,ijk->ij', KguLiQsqrt, KguLiQsqrt)
        
    return mu.T, var.T

def generate_plots(path, name):
    # List dirs in model-dicts
    # For each file model-dicts/path/to/dict save plots to figs/path/to/dict
    param_path = os.path.join(path, name)
    print(param_path)

    gene_path = '/work-zfs/abattle4/karl/GTEx_gene/'
    print(gene_path)

    correlation_path = '/work-zfs/abattle4/karl/correlation_matrices/'
    print(correlation_path)

    gene = name.split('_')[0]
    save_path = path.split('/')
    save_path[4] = 'figs'
    save_path = '/'.join(save_path)
    save_path = os.path.join(save_path, gene)
    print(save_path)

    if not os.path.isdir(save_path):
        print('Making directory {}'.format(save_path))
        os.makedirs(save_path)

    # load data
    names = ['tissue', 'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']

    gene_df = pd.read_csv(gene_path + '{}'.format(gene), sep='\t', names=names)
    r_df = pd.read_csv(correlation_path + '{}'.format(gene), index_col=0)
    idx = ~np.all(np.isnan(r_df.values), axis=0)
    r_df = r_df.iloc[idx, idx]
    r2 = r_df.values ** 2
    
    tissues = np.unique(gene_df.tissue)
    active_variants = r_df.columns.values

    # collect relevant data in tables
    gene_df['log_pval'] = np.log10(gene_df.pval_nominal)
    gene_df['abs_beta'] = np.abs(gene_df.slope)

    beta_df = gene_df.pivot('variant_id', 'tissue', 'abs_beta')
    beta_df = beta_df.loc[active_variants]

    pval_df = gene_df.pivot('variant_id', 'tissue', 'log_pval')
    pval_df = pval_df.loc[active_variants]

    pos = gene_df.pivot('variant_id', 'tissue', 'tss_distance')
    pos = pos.loc[active_variants].iloc[:, 0].values.astype(np.float64)

    in_range = np.logical_and(pos < 500000, pos > -500000)

    # do SVD to get features
    r2_in_range = r2[in_range][:, in_range]
    u, s, vh = np.linalg.svd(r2_in_range)
    X = u * np.sqrt(s)

    # set up inputs/outputs to model, parmaeter initializations
    K = 5
    D = 1000

    Xtrunc = X[:, :D]
    Y = beta_df[in_range].values
    Y = np.clip(Y, 1e-5, None)
    Z = Xtrunc.copy()[::10]

    W = np.random.normal(size=(49, K))
    q_mu_init = np.stack([
        Y.mean(1)[::10] + np.random.normal(size=Y[::10].shape[0]) * 0.1
        for _ in range(K)]).T
    q_mu_init = np.log(np.clip(q_mu_init, Y.mean().min(), None))

    # set up model
    gpflow.reset_default_graph_and_session()
    session = gpflow.get_default_session()

    with gpflow.defer_build():
        feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Z))

        kern_list = [gpflow.kernels.RBF(Xtrunc.shape[1]) for _ in range(K)]
        kernel = mk.SeparateMixedMok(kern_list, W=session.run(tf.nn.sigmoid(W)))
        kernel.W.transform = gpflow.transforms.Log1pe()
        kernel.W.prior = gpflow.priors.Exponential(3.0)

        # initialise mean of variational posterior to be of shape MxL
        q_mu = q_mu_init
        # initialise \sqrt(Σ) of variational posterior to be of shape LxMxM
        q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], K, axis=0) * 1.0

        likelihood = gpflow.likelihoods.Gaussian()
        model = gpflow.models.SVGP(Xtrunc, Y, kernel, likelihood, feat=feature, q_mu=q_mu, q_sqrt=q_sqrt, minibatch_size=50)

    model.compile()

    # load parameters
    param_dict = pickle.load(open(param_path, 'rb'))
    model.assign(param_dict)

    # make prediction
    mu, var = model.predict_f(Xtrunc)
    gmu, gvar = function_means(model, Xtrunc)


    ##########
    # PLOT 1 #
    ##########
    print(save_path + '/W_heatmap.png')            
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(model.kern.W.value, ax=ax[0])
    ax[1].hist(model.kern.W.value)
    plt.legend()
    plt.suptitle(gene)
    plt.savefig(save_path + '/W_heatmap.png')
    plt.close()



    ##########
    # PLOT 2 # 
    ##########
    print(save_path + '/component_scatterplot.png')            
    # plot decomposed components against betas
    fig, axs = plt.subplots(7, 7, figsize=(40, 30), sharex=True, sharey=True)
    for i, ax in enumerate(axs.reshape(-1)): 
        tissue = tissues[i]
        ax.set_title(tissue)

        if np.isin(tissue, beta_df.columns):
            positions = pos[in_range]
            betas = np.abs(beta_df.loc[:, tissue])[in_range]
            pvals = pval_df.loc[:, tissue][in_range]        
            ax.scatter(positions, betas, marker='x', c='k', alpha=0.2)
            for k in range(K):
                w = model.kern.W.value[i, k]
                ax.scatter(positions, w * gmu[:, k], label=str(k))

    plt.suptitle(gene)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/component_scatterplot.png')
    plt.close()



    ##########
    # PLOT 3 #
    ##########
    print(save_path + '/prediction_scatterplot.png')            
    fig, axs = plt.subplots(7, 7, figsize=(40, 30), sharex=True, sharey=True)
    for i, ax in enumerate(axs.reshape(-1)): 
        tissue = tissues[i]
        ax.set_title(tissue)

        if np.isin(tissue, beta_df.columns):
            positions = pos[in_range]
            betas = np.abs(beta_df.loc[:, tissue])[in_range]
            pvals = pval_df.loc[:, tissue][in_range]        
            ax.scatter(positions, betas, marker='x', c='k', alpha=0.2)
            ax.scatter(positions, mu[:, i], c=pvals)

    plt.suptitle(gene)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/prediction_scatterplot.png')
    plt.close()



    ##########
    # PLOT 4 #
    ##########
    print(save_path + '/component_leads.png')            
    # PLOT 3
    # plot each component predictions, colored by r^2 with largest prediction
    fig, ax = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for k in range(K):
        lead = gmu[:, k].argmax()
        color = r2_in_range[lead]
        ax[k].scatter(positions, gmu[:, k], c=color)
        ax[k].axvline(positions[lead])
        ax[k].set_title(r_df.index.values[lead])
        
    plt.suptitle(gene)
    plt.tight_layout()
    plt.savefig(save_path + '/component_leads.png')
    plt.close()



    ##########
    # PLOT 5 #
    ##########
    print(save_path + '/prediction_accuracy_by_pval.png')            
    fig, axs = plt.subplots(7, 7, figsize=(40, 30), sharex=True, sharey=True)
    for i, ax in enumerate(axs.reshape(-1)): 
        tissue = tissues[i]
        ax.set_title(tissue)
        
        if np.isin(tissue, beta_df.columns):
            positions = pos[in_range]
            betas = np.abs(beta_df.loc[:, tissue])[in_range]
            pvals = pval_df.loc[:, tissue][in_range]
            
            ax.scatter(betas, betas, marker='x', c='k', alpha=0.2)
            im = ax.scatter(mu[:, i], betas, c=pvals)
            fig.colorbar(im, ax=ax)
            
    plt.suptitle(gene)
    plt.tight_layout()
    plt.savefig(save_path + '/prediction_accuracy_by_pval.png')
    plt.close()



subdir = sys.argv[1]
path_to_model_dicts = '/work-zfs/abattle4/karl/model-dicts/'
i = 0
for path, subdirs, files in os.walk(path_to_model_dicts):
    for name in files:
        if int(subdir) == i:
            generate_plots(path, name)
        i += 1


