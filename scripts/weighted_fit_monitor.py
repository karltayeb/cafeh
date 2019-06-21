import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys

import scipy as sp
import scipy.spatial

import gpflow
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

from gpflow.likelihoods import Likelihood
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow import transforms, settings, logdensities
import gpflow.training.monitor as mon

import pickle


GENE_PATH = './GTEx_gene/'
CORRELATION_PATH = './correlation_matrices/'
MODEL_SAVE_PATH = './model-dicts/'

names = ['tissue', 'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']
gene = sys.argv[1]
rate = float(sys.argv[2])

run_id = 'weighted/lambda-{}'.format(rate)
run = gene

MODEL_SAVE_PATH = MODEL_SAVE_PATH + run_id + '/'
if not os.path.isdir(MODEL_SAVE_PATH):
    print('Making directory {}'.format(MODEL_SAVE_PATH))
    os.makedirs(MODEL_SAVE_PATH)

MONITOR_SAVE_PATH = './monitor-saves/{}/{}/'.format(run_id, run)
if not os.path.isdir(MONITOR_SAVE_PATH):
    print('Making directory {}'.format(MONITOR_SAVE_PATH))
    os.makedirs(MONITOR_SAVE_PATH)

TBOARD_SAVE_PATH = './model-tensorboard/{}/{}/'.format(run_id, run)
if not os.path.isdir(MONITOR_SAVE_PATH):
    print('Making directory {}'.format(TBOARD_SAVE_PATH))
    os.makedirs(TBOARD_SAVE_PATH)


class WeightedGaussian(Likelihood):
    """
    same as gaussian likelihood except last column of Y has weights
    """
    def __init__(self, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def logp(self, F, Y):
        Y = Y[:, ::2]
        W = Y[:, 1::2]
        return logdensities.gaussian(Y, F, self.variance*W)

    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        Y = Y[:, ::2]
        W = Y[:, 1::2]
        return logdensities.gaussian(Y, Fmu, Fvar + self.variance * W)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        A = Y[:, ::2]
        B = Y[:, 1::2]
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance * B) \
               - 0.5 * (tf.square(A - Fmu) + Fvar) / (self.variance * B)


class SaveModelDict(mon.BaseTensorBoardTask):
    def __init__(self, file_writer, file_path, model):
        super().__init__(file_writer, model)
        self.file_path = file_path
    
    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        pickle.dump(self.model.read_values(), open(self.file_path, 'wb'))


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

se_df = gene_df.pivot('variant_id', 'tissue', 'slope_se')
se_df = se_df.loc[active_variants]

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
minibatch_size = 50

# get embedding
Xtrunc = X[:, :D]

# get data + standard errors for weighting
Y = np.zeros((in_range.sum(), 2 * tissues.size))
betas = np.clip(beta_df[in_range].values, 1e-8, None)
Y[:, ::2] = betas

variances = se_df[in_range].values ** 2
variances[np.isnan(variances)] = 10000
variances = np.clip(variances, 1e-5, None)
Y[:, 1::2] = variances

# initialize variational parameters
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
    kernel = mk.SeparateMixedMok(kern_list, W=W)
    kernel.W.transform = gpflow.transforms.Log1pe()
    kernel.W.prior = gpflow.priors.Exponential(rate)
    kernel.W = W

    # initialise mean of variational posterior to be of shape MxL
    q_mu = q_mu_init
    # initialise \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(Z.shape[0])[None, ...], K, axis=0) * 1.0
    
    likelihood = WeightedGaussian()
    m = gpflow.models.SVGP(Xtrunc, Y, kernel, likelihood, feat=feature, q_mu=q_mu, q_sqrt=q_sqrt, num_latent=tissues.size, minibatch_size=50)
    m.q_mu.transform = gpflow.transform.Lop1pe()
    m.q_mu = np.clip(q_mu_init, 1e-8, None)
    
m.compile()


session = m.enquire_session()
global_step = mon.create_global_step(session)

print_task = mon.PrintTimingsTask().with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(100))\
    .with_exit_condition(True)

sleep_task = mon.SleepTask(0.01).with_name('sleep').with_name('sleep')

saver_task = mon.CheckpointTask('./monitor-saves/{}/{}/'.format(run_id, run)).with_name('saver')\
    .with_condition(mon.PeriodicIterationCondition(200))\
    .with_exit_condition(True)

file_writer = mon.LogdirWriter('./model-tensorboard/{}/{}/'.format(run_id, run))

model_tboard_task = mon.ModelToTensorBoardTask(file_writer, m).with_name('model_tboard')\
    .with_condition(mon.PeriodicIterationCondition(10))\
    .with_exit_condition(True)

lml_tboard_task = mon.LmlToTensorBoardTask(file_writer, m).with_name('lml_tboard')\
    .with_condition(mon.PeriodicIterationCondition(100))\
    .with_exit_condition(True)

path_to_save = MODEL_SAVE_PATH + '{}_param_dict'.format(gene)
savemodeldict = SaveModelDict(file_writer, path_to_save, m).with_name('save_model_dict')\
    .with_condition(mon.PeriodicIterationCondition(200))\
    .with_exit_condition(True)

monitor_tasks = [print_task, model_tboard_task, lml_tboard_task, saver_task, savemodeldict, sleep_task]
monitor = mon.Monitor(monitor_tasks, session, global_step)

optimiser = gpflow.train.AdamOptimizer(0.01)

with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
    optimiser.minimize(m, step_callback=monitor, maxiter=20000, global_step=global_step)

file_writer.close()

m.anchor(session)
with open(path_to_save, 'wb') as f:
    pickle.dump(m.read_values(), f)

