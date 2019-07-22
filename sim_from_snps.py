import marginal_models
import autograd_transforms
from training_and_plotting import train

import autograd.numpy as np
import autograd.scipy as sp
import pandas as pd
import pickle

M, T, = 1, 10

GENE_PATH = './top_genes/'
CORRELATION_PATH = './data/correlations/'
MODEL_SAVE_PATH = './output/'
ITERS = 10000

names = ['tissue', 'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']
gene = 'ENSG00000101307.15'

gene_df = pd.read_csv(GENE_PATH + '{}'.format(gene), sep='\t', names=names)
r_df = pd.read_csv(CORRELATION_PATH + '{}'.format(gene), index_col=0)

idx = ~np.all(np.isnan(r_df.values), axis=0)
r_df = r_df.iloc[idx, idx]

tissues = np.unique(gene_df.tissue)
active_variants = r_df.columns.values

pos = gene_df.pivot('variant_id', 'tissue', 'tss_distance')
pos = pos.loc[active_variants].iloc[:, 0].values.astype(np.float64)

in_range = np.logical_and(pos < 100000, pos > -100000)

Sigma = r_df.values[in_range][:, in_range]

u, s, vh = np.linalg.svd(Sigma)
X = (u * np.sqrt(s))[:, :900]

def generate_data(Sigma, num_causal=1, effect_size=1, num_tissues=1):
    N = Sigma.shape[0]
    causal_status = np.zeros(N)
    causal_status[np.random.choice(N, num_causal)] = effect_size
    expected_effects = Sigma @ causal_status[:, None]
    
    L = np.linalg.cholesky(Sigma + np.eye(N) * 1e-6)
    observed_effects = L @ np.random.normal(size=(N, num_tissues)) + expected_effects
    return causal_status, expected_effects, observed_effects

data_dict = {}
save_dict = {}
marginal_functions = [x for x in dir(marginal_models) if 'functions' in x]

for run in range(3):
    causal, expected, Y = generate_data(Sigma, num_causal=1, effect_size=1, num_tissues=T)
    data_dict[run] = (causal, expected, Y)

    for function in marginal_functions:
        print(function)
        for transform in ['normalize_transform']:
            if 'independent' in function:
                for kwargs in [{'train_A': False}, {'train_A': True}]:
                    try:
                        fname = '_'.join(function.split('_')[:-1])
                        tname = '_'.join(transform.split('_')[:-1])
                        key = '{}_{}_{}_{}_{}_{}'.format(fname, tname, M, T, kwargs['train_A'], run)

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
                    key = '{}_{}_{}_{}_{}_{}'.format(fname, tname, M, T, False, run)


                    print('\t{}'.format(key))
                    funcs = getattr(marginal_models, function)
                    trans = getattr(autograd_transforms, transform)
                    Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, verbose=False)
                    save_dict[key] = (Z, Z_init, params)
                except:
                    print('\t!!!!!!!')

# save save_dict
save_name = '{}_M{}_T{}_ouputs'.format(gene, M, T)
pickle.dump(save_dict, open(save_name, 'wb'))

data_save_name = '{}_M{}_T{}_data'.format(gene, M, T)
pickle.dump(data_dict, open(data_save_name, 'wb'))