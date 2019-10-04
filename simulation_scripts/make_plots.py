import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_g(gmu, W, bottom=None, top=None, thresh=None, sharey=False, title=''):
    N, Q = gmu.shape
    active = np.arange(Q)

    if thresh is not None:
        active = active[W.max(0) > thresh]

    fig, ax = plt.subplots(1, active.size, figsize=(active.size*4, 3), sharey=sharey)
    for i, component in enumerate(active):
        ax[i].scatter(np.arange(N), gmu[:, component], alpha=0.2)
        ax[i].set_xlabel('SNP')
        ax[i].set_title('Component {} {}'.format(component, title))

        if bottom is not None:
            ax[i].scatter(bottom[component], np.ones(bottom[component].shape[0])*gmu[:, component].min(), marker='|', c='k')

        if top is not None:
            ax[i].scatter(top, np.ones(top.size)*gmu[:,component].max(), marker='*', c='r', s=100)

    ax[0].set_ylabel('component_mean')
    return fig

run_ids = os.listdir('./T10_simulation_output/')

Sigma, causal_snps, tissue_membership, causal = pickle.load(
    open('./T10_simulation', 'rb'))

for run_id in run_ids:
    if 'effect' in run_id:
        continue
    print(run_id)
    data_dict = pickle.load(
        open('./T10_simulation_output/{}'.format(run_id), 'rb'))

    effectsize = float(run_id.split('_')[0][2:])
    print(effectsize)

    q_gmu, q_gvar, W, indices = data_dict['q_gmu'], data_dict['q_gvar'], data_dict['W'], data_dict['local_indices']
    # Y = data_dict['Y']


    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(causal[:, causal_snps], cmap='Blues', square=True, ax=ax[1], cbar=False)
    sns.heatmap(W, cmap='Blues', square=True, ax=ax[0])
    ax[0].set_title('W')
    ax[1].set_title('Tissue x Causal SNP')
    ax[0].set_ylabel('Tissue')
    ax[0].set_xlabel('Component')
    ax[1].set_xlabel('Causal SNP')
    plt.tight_layout()
    plt.savefig('./T10_simulation_figs/{}_heatmap.png'.format(run_id))
    plt.close()


    plot_g(q_gmu, W, bottom=indices, top=causal_snps, title='Mean')
    plt.savefig('./T10_simulation_figs/{}_means.png'.format(run_id))
    plt.close()

    plot_g(np.array([np.diag(x) for x in q_gvar]).T, W, bottom=indices, top=causal_snps, sharey=True, title='Variance')
    plt.savefig('./T10_simulation_figs/{}_variance.png'.format(run_id))
    plt.close()

    T = causal.shape[0]
    fig, ax = plt.subplots(1, T, figsize=(3*T, 3), sharex=True, sharey=True)

    for t in range(T):
        ax[t].scatter((q_gmu@W.T)[:, t], (effectsize * Sigma @ causal.T)[:, t], alpha=0.3, c='k', marker='x')
        ax[t].set_title('Tissue {}'.format(t))
        ax[t].set_xlabel('Mean')
    ax[0].set_ylabel('Data')
        
    plt.suptitle('Reconstruction (Noise Free)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./T10_simulation_figs/{}_reconstruction.png'.format(run_id))
