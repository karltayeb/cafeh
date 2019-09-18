import pickle
import sys
sys.path.append("..")


import numpy as np
import scipy as sp

from linear_mvn import update_W_map, update_W_se, update_W
from linear_mvn import update_variational_params_inducing
from linear_mvn import MultipleLinearKernel, _make_covariances
from sklearn.cluster import SpectralClustering
from itertools import product

idx = int(sys.argv[1])

effectsizes = [5, 10]
kernel_types = ['fic', 'linear']
local_boolean = [True]
regularize_Sigma = [0.1, 1.0]
W_updates = ['se', 'map', 'full']
penalties = [10]
num_components = [10]

states = list(product(effectsizes, kernel_types, local_boolean, regularize_Sigma, W_updates, penalties, num_components))
effectsize, kernel_type, local_inducing, reg, W_update_type, penalty, Q = \
    states[idx]

save_name = 'e-{}_Q-{}_local-{}_reg-{}_{}_{}_lambda-{}'.format(effectsize, Q, local_inducing, regularize_Sigma, kernel_type, W_update_type, penalty)
print(save_name)

if W_update_type == 'se':
    W_update = update_W_se
elif W_update_type == 'map':
    W_update = update_W_map
else:
    W_update = update_W


# load simulations
Sigma, causal_snps, tissue_membership, causal = pickle.load(
    open('T10_simulation', 'rb'))
T, N = causal.shape

# make embedding
u, s, vh = np.linalg.svd(Sigma)
X = (u * np.sqrt(s))
D = np.arange(N)[np.isclose(s, 0)][0]
X = X[:, :D]


# cluster SNPs, make indices
R2 = Sigma**2
clustering = SpectralClustering(
    n_clusters=Q, affinity='precomputed', assign_labels="discretize").fit(R2)
local_indices = [np.arange(N)[clustering.labels_ == i] for i in range(Q)]
if local_inducing:
    Zs = [X[idx] if len(idx) > 0 else X[::100] for idx in local_indices]
else:
    Zs = [np.eye(D) for _ in range(Q)]

# set up kernel, initialize parameters
print(kernel_type)
kernel = MultipleLinearKernel(np.random.random((T, Q))*5, Zs, kernel=kernel_type)
precompute = _make_covariances(kernel, X, Zs)

q_gmu = np.zeros((N, Q))
q_gvar = np.repeat(np.eye(N)[None, ...], Q, axis=0) * 1.0

q_gmu_z = [np.zeros((Z.shape[0])) for Z in Zs]
q_gvar_z = [np.eye(Z.shape[0]) for Z in Zs]

# generate data
Y = effectsize * Sigma @ causal.T + np.linalg.cholesky(Sigma + np.eye(N)*1e-6) @ np.random.normal(size=causal.T.shape)

S = (Sigma + np.eye(N)*reg) / (1 + reg)
for _ in range(100):
    print('updating variational params')
    q_gmu_z, q_gvar_z, q_gmu, q_gvar = update_variational_params_inducing(
        kernel.W, precompute, Y, Sigma, q_gmu_z, q_gvar_z, niter=10)

    print('updating W')
    kernel.W = np.array(W_update(Y, q_gmu, q_gvar, Sigma, penalty=penalty))

    data_dict = {
        'q_gmu': q_gmu,
        'q_gvar': q_gvar,
        'W': kernel.W,
        'Y': Y,
        'local_indices': local_indices
    }
    pickle.dump(data_dict, open(save_name, 'wb'))
