import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import autograd
import autograd.numpy as np
import autograd.scipy as sp

import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn

from autograd import value_and_grad
from scipy.optimize import minimize

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from autograd_transforms import identity_transform, spherical_transform, normalize_transform

import marginal_models
import joint_models
import autograd_transforms

from training_and_plotting import *
import pickle
import sys

N = 100
M = 1
p = 0.9
T = int(sys.argv[1])

save_dict = {}
data_dict = {}

joint_functions = [x for x in dir(joint_models) if 'functions' in x]
marginal_functions = [x for x in dir(marginal_models) if 'functions' in x]

for run in range(2):
    # generate data
    G, X_full, Y_full, Sigma_full = generate_data(N, 1000, 0.5, p, T)
    data_dict[run] = (G, X_full, Y_full, Sigma_full)
    
    for a in [0, 5, 10, 25]:
        to_delete = list(range(int(N/2) - a, int(N/2)+a))
        X = np.delete(X_full, to_delete, axis=0)
        Y = np.delete(Y_full, to_delete, axis=0)
        Sigma = np.delete(np.delete(Sigma_full, to_delete, axis=0), to_delete, axis=1)

        for function in joint_functions:
            print(function)
            for transform in ['spherical_transform', 'normalize_transform']:
                for kwargs in [{'train_v': False}, {'train_v': True}]:
                    try:
                        fname = '_'.join(function.split('_')[:-1])
                        tname = '_'.join(transform.split('_')[:-1])
                        key = '{}_{}_{}_{}_{}_missing{}_{}'.format(fname, tname, M, T, kwargs['train_v'], a, run)

                        print('\t{}'.format(key))
                        funcs = getattr(joint_models, function)
                        trans = getattr(autograd_transforms, transform)
                        Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, kwargs=kwargs, verbose=False)
                        save_dict[key] = (Z, Z_init, params)
                    except:
                        print('\t!!!!!!!')

        for function in marginal_functions:
            print(function)
            for transform in ['spherical_transform', 'normalize_transform']:
                if 'independent' in function:
                    for kwargs in [{'train_A': False}, {'train_A': True}]:
                        try:
                            fname = '_'.join(function.split('_')[:-1])
                            tname = '_'.join(transform.split('_')[:-1])
                            key = '{}_{}_{}_{}_{}_missing{}_{}'.format(fname, tname, M, T, kwargs['train_A'], a, run)

                            print('\t{}'.format(key))
                            funcs = getattr(marginal_models, function)
                            trans = getattr(autograd_transforms, transform)
                            Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, kwargs=kwargs, verbose=False)
                            save_dict[key] = (Z, Z_init, params)
                        except:
                            print('\t!!!!!!!')
                else:
                    try:
                        fname = '_'.join(function.split('_')[:-1])
                        tname = '_'.join(transform.split('_')[:-1])
                        key = '{}_{}_{}_{}_{}_missing{}_{}'.format(fname, tname, M, T, False, a, run)

                        print('\t{}'.format(key))
                        funcs = getattr(marginal_models, function)
                        trans = getattr(autograd_transforms, transform)
                        Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, verbose=False)
                        save_dict[key] = (Z, Z_init, params)
                    except:
                        print('\t!!!!!!!')


# save save_dict
save_name = 'N{}_M{}_p{}_T{}_missing_ouputs'.format(N, M, p, T)
pickle.dump(save_dict, open(save_name, 'wb'))

data_save_name = 'N{}_M{}_p{}_T{}_missing_data'.format(N, M, p, T)
pickle.dump(data_dict, open(data_save_name, 'wb'))