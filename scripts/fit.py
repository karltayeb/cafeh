import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys

import scipy as sp
import scipy.spatial

import gpflow
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

import pickle


GENE_PATH = '~/GTEx_gene/'
CORRELATION_PATH = '~/correlation_matrices/'

MODEL_SAVE_PATH = '~/model_dict_saves/'

names = ['tissue', 'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']
gene = sys.argv[1]

gene_df = pd.read_csv(GENE_PATH + '{}'.format(gene), sep='\t', names=names)
r_df = pd.read_csv(CORRELATION_PATH + '{}'.format(gene), index_col=0)

idx = ~np.all(np.isnan(r_df.values), axis=0)
r_df = r_df.iloc[idx, idx]

tissues = np.unique(gene_df.tissue)

r2 = r_df.values ** 2

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
    kernel.W.prior = gpflow.priors.Laplace(0, 0.5)

    # initialise mean of variational posterior to be of shape MxL
    q_mu = q_mu_init
    # initialise \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], K, axis=0) * 1.0
    
    likelihood = gpflow.likelihoods.Gaussian()
    model = gpflow.models.SVGP(Xtrunc, Y, kernel, likelihood, feat=feature, q_mu=q_mu, q_sqrt=q_sqrt)
        
model.compile()

if not os.path.isdir(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

for i in range(50):
    print('{}: {}'.format(i, model.compute_log_likelihood())
    opt = gpflow.training.ScipyOptimizer()
    opt.minimize(model, maxiter=1000)

    pickle.dump(model.read_values(), open(MODEL_SAVE_PATH+'{}_model'.format(gene), 'wb'))