import marginal_models
import autograd_transforms
from training_and_plotting import train

import autograd.numpy as np
import autograd.scipy as sp
import pandas as pd
import sys
import pickle

M, T, function, transform, train_params = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], bool(int(sys.argv[5]))

GENE_PATH = '../GTEx_gene/'
CORRELATION_PATH = '../correlation_matrices/'
ITERS = 5000

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

ldness = np.mean(np.sort(np.abs(Sigma), 1)[:, -11:-1], 1)
causal_snps = np.unique([np.abs(ldness - target_ld).argmin() for target_ld in np.linspace(0, 1, 11)])

def generate_data(Sigma, causal_snps=None, effect_size=1, num_tissues=1):
    N = Sigma.shape[0]
    causal_status = np.zeros(N)

    if causal_snps is None:
        causal_snps = np.random.choice(N, 1)
    causal_status[causal_snps] = effect_size
    expected_effects = Sigma @ causal_status[:, None]

    L = np.linalg.cholesky(Sigma + np.eye(N) * 1e-6)
    observed_effects = L @ np.random.normal(size=(N, num_tissues)) + expected_effects
    return causal_status, expected_effects, observed_effects

data_dict = {}
save_dict = {}
marginal_functions = [x for x in dir(marginal_models) if 'functions' in x]


fname = '_'.join(function.split('_')[:-1])
tname = '_'.join(transform.split('_')[:-1])

funcs = getattr(marginal_models, function)
trans = getattr(autograd_transforms, transform)

for run in range(100):
    data_dict[run] = {}
    for causal_snp in causal_snps:
        causal, expected, Y = generate_data(Sigma, causal_snps=causal_snp, effect_size=1, num_tissues=T)
        data_dict[run][causal_snp] = (causal, expected, Y)
        print(function)
        if 'independent' in function:
            kwargs = {'train_A': train_params}
            key = '{}_{}_{}_{}_{}_{}_{}'.format(fname, tname, M, T, kwargs['train_A'], causal_snp, run)
            print('\t{}'.format(key))
            Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, kwargs=kwargs, verbose=False)
            save_dict[key] = (Z, Z_init, params)
        else:
            try:
                key = '{}_{}_{}_{}_{}_{}_{}'.format(fname, tname, M, T, False, causal_snp, run)
                print('\t{}'.format(key))
                Z, Z_init, params = train(funcs, X, Y, Sigma, M=M, transform=trans, verbose=False)
                save_dict[key] = (Z, Z_init, params)
            except:
                print('\t!!!!!!!')

    if (run % 10) == 0:
        # save save_dict
        save_name = '{}_M{}_T{}_{}_{}_{}_ouputs'.format(gene, M, T, fname, tname, train_params)
        pickle.dump(save_dict, open(save_name, 'wb'))

        data_save_name = '{}_M{}_T{}_{}_{}_{}_data'.format(gene, M, T, fname, tname, train_params)
        pickle.dump(data_dict, open(data_save_name, 'wb'))

# save save_dict
save_name = '../simulation_from_snps/{}_M{}_T{}_{}_{}_{}_ouputs'.format(gene, M, T, fname, tname, train_params)
pickle.dump(save_dict, open(save_name, 'wb'))

data_save_name = '../simulation_from_snps/{}_M{}_T{}_{}_{}_{}_data'.format(gene, M, T, fname, tname, train_params)
pickle.dump(data_dict, open(data_save_name, 'wb'))
