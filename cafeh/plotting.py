import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from .kls import unit_normal_kl, normal_kl, categorical_kl
import pickle

######################
# PLOTTING FUNCTIONS #
######################

def plot_assignment_kl(self, thresh=0.5, save_path=None, show=True):
    """
    Plot similarity between causal components
    If CAFEH is run without estimating effect sizes, and prior_variance is too small
    you will see redundancy in the components.
    """
    kls = np.zeros((self.dims['K'], self.dims['K']))
    for k1 in range(self.dims['K']):
        for k2 in range(self.dims['K']):
            kls[k1, k2] = categorical_kl(self.pi.T[:, k1], self.pi.T[:, k2])
    active = np.any(self.active > thresh, 0)
    sns.heatmap(kls[active][:, active], cmap='Blues', xticklabels=np.arange(self.dims['K'])[active], yticklabels=np.arange(self.dims['K'])[active])
    plt.title('Pairwise KL of Component Multinomials')
    plt.xlabel('Component')
    plt.ylabel('Component')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_credible_sets_ld(self, snps=None, alpha=0.9, thresh=0.5, save_path=None, show=True):
    """
    Heatmap of LD among `alpha` credible sets with `p_active > thresh` in atleast one study
    can also include additional snps via snps argument
    """
    if snps is None:
        snps = []
    credible_sets, purity = self.get_credible_sets(alpha=alpha)
    for k in np.arange(self.dims['K']):
        if purity[k] > thresh:
            snps.append(credible_sets[k])

    sizes = np.array([x.size for x in snps])
    snps = np.concatenate(snps)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    ld = self.get_ld(snps=snps)
    if np.ndim(ld) == 2:
        ld = ld ** 2
    else:
        ld = (ld ** 2).min(0)
    sns.heatmap(
        ld, cmap='RdBu_r', center=0, ax=ax,
        square=True, annot=False, cbar=True,
        yticklabels=self.snp_ids[snps], xticklabels=[]
    )
    ax.hlines(np.cumsum(sizes), *ax.get_xlim(), colors='w', lw=3)
    ax.vlines(np.cumsum(sizes), *ax.get_ylim(), colors='w', lw=3)
    plt.title('alpha={} confidence set LD'.format(alpha))
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    

def plot_components(self, thresh=0.5, save_path=None, show=True):
    """
    plot inducing point posteriors, weight means, and probabilities
    """

    weights = self.get_expected_weights()
    active_components = self.active.max(0) > 0.5
    """
    cs, pur = self.get_credible_sets()
    active_components = np.array([k for k in range(self.dims['K']) if pur[k] >= thresh])
    if active_components.size == 0:
        active_components = np.array([0])
    """

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for k in np.arange(self.dims['K'])[active_components]:
        ax[2].scatter(
            np.arange(self.dims['N'])[self.pi.T[:, k] > 2/self.dims['N']],
            self.pi.T[:, k][self.pi.T[:, k] > 2/self.dims['N']],
            alpha=0.5, label='k{}'.format(k))
    ax[2].scatter(np.arange(self.dims['N']), np.zeros(self.dims['N']), alpha=0.0)
    ax[2].set_title('pi')
    ax[2].set_xlabel('SNP')
    ax[2].set_ylabel('probability')
    ax[2].legend(bbox_to_anchor=(1.04,1), loc="upper left")


    sns.heatmap(weights[:, active_components], annot=False, cmap='RdBu_r', ax=ax[1], yticklabels=[], center=0)
    ax[1].set_title('weights')
    ax[1].set_xlabel('component')

    sns.heatmap((self.active)[:, active_components],
        annot=False, cmap='Blues', ax=ax[0],
        vmin=0, vmax=1, yticklabels=self.study_ids)
    ax[0].set_title('active')
    ax[0].set_xlabel('component')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
