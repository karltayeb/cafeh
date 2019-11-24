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
    kls = np.zeros((self.K, self.K))
    for k1 in range(self.K):
        for k2 in range(self.K):
            kls[k1, k2] = categorical_kl(self.pi[:, k1], self.pi[:, k2])
    active = np.any(self.active > thresh, 0)
    sns.heatmap(kls[active][:, active], cmap='Blues', xticklabels=np.arange(self.K)[active], yticklabels=np.arange(self.K)[active])
    plt.title('Pairwise KL of Component Multinomials')
    plt.xlabel('Component')
    plt.ylabel('Component')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_component_correlations(self, save_path=None):
    sns.heatmap(np.abs(np.corrcoef((self.X @ self.pi).T)), cmap='Reds')
    plt.title('Component correlation')
    plt.xlabel('Component')
    plt.ylabel('Component')
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_component_x_component(self, save_path=None, show=True):
    active = np.any(self.active > 0.5, axis=0)
    components = (self.X @ self.pi)[:, active]
    num_active = active.sum()
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    fig, ax = plt.subplots(num_active + 1, num_active + 1, figsize=(4 * (num_active+1), 4 * (num_active+1)))
    for i in range(num_active):
        ax[i + 1, 0].scatter(pos, components[:, i])
        ax[i + 1, 0].set_ylabel('Component {}'.format(i))
        ax[0, i + 1].scatter(pos, components[:, i])
        ax[0, i + 1].set_title('Component {}'.format(i))

    for i in range(num_active):
        for j in range(num_active):
            if i == j:
                ax[i+1, j+1].scatter(components[:, i], components[:, j], marker='x', alpha=0.5, c='k')
            else:
                ax[i+1, j+1].scatter(components[:, i], components[:, j], marker='x', alpha=0.5, c='r')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_credible_sets_ld(self, snps=None, alpha=0.9, thresh=0.5, save_path=None, show=True):
    if snps is None:
        snps = []
    active = self.active.max(0) > thresh
    for k in np.arange(self.K)[active]:
        cset_size = (np.cumsum(np.flip(np.sort(self.pi[:, k]))) < alpha).sum() + 1
        cset = np.flip(np.argsort(self.pi[:, k])[-cset_size:])
        snps.append(cset)

    sizes = np.array([x.size for x in snps])

    snps = np.concatenate(snps)
    fig, ax = plt.subplots(1, figsize=(6, 5))
    sns.heatmap(self.X[snps][:, snps],
        cmap='RdBu_r', vmin=-1, vmax=1, ax=ax, square=True, annot=False, cbar=True,
        yticklabels=self.snp_ids[snps], xticklabels=[])
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
    if self.weights.ndim == 2:
        weights = self.weights * self.active
    else:
        weights = np.einsum('ijk,kj->ij', self.weights, self.pi)
    # make plot
    active_components = self.active.max(0) > thresh
    if np.all(~active_components):
        active_components[:3] = True

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for k in np.arange(self.K)[active_components]:
        if (self.pi[:, k] > 2/self.N).sum() > 0:
            ax[2].scatter(np.arange(self.N)[self.pi[:, k] > 2/self.N], self.pi[:, k][self.pi[:, k] > 2/self.N], alpha=0.5, label='k{}'.format(k))
    ax[2].scatter(np.arange(self.N), np.zeros(self.N), alpha=0.0)
    ax[2].set_title('pi')
    ax[2].set_xlabel('SNP')
    ax[2].set_ylabel('probability')
    ax[2].legend()

    vmax = np.abs(weights[:, active_components]).max()
    vmin = -vmax
    sns.heatmap(weights[:, active_components], annot=False, cmap='RdBu_r', ax=ax[1], yticklabels=[], vmin=vmin, vmax=vmax)
    ax[1].set_title('weights')
    ax[1].set_xlabel('component')

    sns.heatmap((self.active)[:, active_components],
        annot=False, cmap='Blues', ax=ax[0],
        vmin=0, vmax=1, yticklabels=self.tissue_ids)
    ax[0].set_title('active')
    ax[0].set_xlabel('component')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_decomposed_manhattan(self, tissues=None, components=None, save_path=None, show=True):
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    W = self.active * self.weights
    c = (self.X @ self.pi)

    pred = self._compute_prediction()
    logp = -norm.logcdf(-np.abs(pred)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    fig, ax = plt.subplots(2, tissues.size, figsize=((tissues.size)*4, 6), sharey=False)
    for i, t in enumerate(tissues):
        ulim = []
        llim = []
        ax[0, i].set_title('{}\n-log p'.format(self.tissue_ids[t]))
        ax[1, i].set_title('components')
        ax[0, 0].set_title('-log p')

        ax[0, i].scatter(pos, logp[t], marker='x', c='k', alpha=0.5)
        ax[0, 0].set_title('-log p')
        ax[1, 0].set_title('- log p')
        ulim.append(logp[t].max())
        llim.append(logp[t].min())

        ulim = []
        llim = []
        for k in components:
            predk = self._compute_prediction() - self._compute_prediction(k=k)
            logpk = -norm.logcdf(-np.abs(predk)) - np.log(2)

            if i == 0:
                ax[1, i].scatter(pos, logpk, marker='o', alpha=0.5, label='k{}'.format(k))
            else:
                ax[1, i].scatter(pos, logpk, marker='o', alpha=0.5)
            ulim.append(logpk.max())
            llim.append(logpk.min())

        ulim = np.array(ulim).max()
        llim = np.array(llim).min()

        #ax[0, i].set_ylim(llim, ulim)
        #ax[1, i].set_ylim(llim, ulim)
        ax[1, i].set_xlabel('SNP position')
        fig.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_decomposed_manhattan2(self, tissues=None, width=None, components=None, save_path=None):
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    if width is None:
        width = int(np.sqrt(tissues.size)) + 1
        height = width
    else:
        height = int(tissues.size / width) + 1

    pred = ((self.active * self.weights) @ (self.X @ self.pi).T)
    logp = - norm.logcdf(-np.abs(pred)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    W = self.active * self.weights
    c = (self.X @ self.pi)

    pred = self._compute_prediction()
    fig, ax = plt.subplots(height, width, figsize=(width*4, height*3), sharey=False)

    ax = np.array(ax).flatten()
    for i, t in enumerate(tissues):
        ax[i].set_title('{}\nby component'.format(self.tissue_ids[t]))

        for k in components:
            predk = self._compute_prediction() - self._compute_prediction(k=k)
            logpk = -norm.logcdf(-np.abs(predk)) - np.log(2)
            if i == 0:
                ax[i].scatter(pos, logpk, marker='o', alpha=0.5, label='k{}'.format(k))
            else:
                ax[i].scatter(pos, logpk, marker='o', alpha=0.5)
        ax[i].set_xlabel('SNP position')
        fig.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_decomposed_zscores(self, tissues=None, components=None, save_path=None, show=True):
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    pred = self._compute_prediction()
    logp = -np.log(norm.cdf(-np.abs(pred))*2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    pred = self._compute_prediction()
    fig, ax = plt.subplots(2, tissues.size, figsize=((tissues.size)*4, 6), sharey=False)
    for i, t in enumerate(tissues):
        ulim = []
        llim = []
        ax[0, i].set_title('{}\nzscores'.format(self.tissue_ids[t]))
        ax[1, i].set_title('components')

        ax[0, i].scatter(pos, pred[t], marker='x', c='k', alpha=0.5)
        ulim.append(pred[t].max())
        llim.append(pred[t].min())

        for k in components:
            predk = self._compute_prediction()[t] - self._compute_prediction(k=k)[t]
            if i == 0:
                ax[1, i].scatter(pos, predk, marker='o', alpha=self.active[t, k], label='k{}'.format(k))
            else:
                ax[1, i].scatter(pos, predk, marker='o', alpha=self.active[t, k])
            ulim.append(predk.max())
            llim.append(predk.min())

        ulim = np.array(ulim).max()
        llim = np.array(llim).min()

        #ax[0, i].set_ylim(llim, ulim)
        #ax[1, i].set_ylim(llim, ulim)
        ax[1, i].set_xlabel('SNP position')
        fig.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_decomposed_zscores2(self, tissues=None, width=None, components=None, save_path=None):
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    if width is None:
        width = int(np.sqrt(tissues.size)) + 1
        height = width
    else:
        height = int(tissues.size / width) + 1

    pred = self._compute_prediction()
    logp = -np.log(norm.cdf(-np.abs(pred))*2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    W = self.active * self.weights
    c = (self.X @ self.pi)

    pred = W @ c.T
    fig, ax = plt.subplots(height, width, figsize=(width*4, height*3), sharey=False)

    ax = np.array(ax).flatten()
    for i, t in enumerate(tissues):
        ax[i].set_title('{}\nby component'.format(self.tissue_ids[t]))

        for k in components:
            predk = self._compute_prediction() - self._compute_prediction(k=k)
            if i == 0:
                ax[i].scatter(pos, predk, marker='o', alpha=0.5, label='k{}'.format(k))
            else:
                ax[i].scatter(pos, predk, marker='o', alpha=0.5)
        ax[i].set_xlabel('SNP position')
        fig.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_residual_zscores(self, tissues=None, components=None, save_path=None):
    """
    plot residual of tissue t with components removed
    """
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    W = self.active * self.weights
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    fig, ax = plt.subplots(components.size, tissues.size, figsize=(tissues.size*4, components.size*3), sharey=False)
    residual = self._compute_residual()
    for j, k in enumerate(components):
        for i, t in enumerate(tissues):
            residual_k = self._compute_residual(k=k)
            #ax[j, i].scatter(pos, self.Y[t], alpha=0.5, marker='x', color='k')
            #ax[j, i].scatter(pos, residual, alpha=0.5, marker='o', color='r')
            line = np.linspace(self.Y[t].min(), self.Y[t].max(), 10)
            ax[j, i].plot(line, line, c='b')
            ax[j, i].scatter(self.Y[t], residual[t], alpha=0.3, marker='x', color='k', label='full residual')

            if self.active[t, k] > 0.5:
                ax[j, i].scatter(self.Y[t], residual_k, alpha=0.5, marker='o', color='g', label='active component residual')
            else:
                ax[j, i].scatter(self.Y[t], residual_k, alpha=0.5, marker='o', color='r', label='inactive component residual')

            ax[j, i].set_title('{}\n{} removed'.format(self.tissue_ids[t], k))
            ax[j, i].set_xlabel('observed z score')
            ax[j, i].set_ylabel('residual z score')
            ax[j, i].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_residual_manhattan(self, tissues=None, components=None, save_path=None):
    """
    plot residual of tissue t with components removed
    """
    if tissues is None:
        tissues = np.arange(self.T)
    else:
        tissues = np.arange(self.T)[np.isin(self.tissue_ids, tissues)]

    if components is None:
        components = np.arange(self.K)[np.any((self.active > 0.5), 0)]

    W = self.active * self.weights
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])
    logp = -norm.logcdf(-np.abs(self.Y)) - np.log(2)

    fig, ax = plt.subplots(components.size, tissues.size, figsize=(tissues.size*4, components.size*3), sharey=False)

    residual = self._compute_residual()
    residual_logp = -norm.logcdf(-np.abs(residual)) - np.log(2)

    for j, k in enumerate(components):
        for i, t in enumerate(tissues):
            residual_k = self._compute_residual(k=k)
            residual_k_logp = -norm.logcdf(-np.abs(residual_k)) - np.log(2)

            #ax[j, i].scatter(pos, self.Y[t], alpha=0.5, marker='x', color='k')
            #ax[j, i].scatter(pos, residual, alpha=0.5, marker='o', color='r')
            line = np.linspace(logp[t].min(), logp[t].max(), 10)
            ax[j, i].plot(line, line, c='b')
            ax[j, i].scatter(logp[t], residual_logp[t], alpha=0.3, marker='x', color='k', label='full residual')

            if self.active[t, k] > 0.5:
                ax[j, i].scatter(logp[t], residual_k_logp, alpha=0.5, marker='o', color='g', label='active component residual')
            else:
                ax[j, i].scatter(logp[t], residual_k_logp, alpha=0.5, marker='o', color='r', label='inactive component residual')

            ax[j, i].set_title('{}\n{} removed'.format(self.tissue_ids[t], k))
            ax[j, i].set_xlabel('observed -log pvalue')
            ax[j, i].set_ylabel('residual -log pvalue')
            ax[j, i].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_predictions(self, save_path=None, show=True):
    """
    plot predictions against observed z scores
    """
    pred = self._compute_prediction()
    fig, ax = plt.subplots(2, self.T, figsize=(4*self.T, 6), sharey=True)
    for t in range(self.T):
        ax[0, t].scatter(np.arange(self.N), self.Y[t], marker='x', c='k', alpha=0.5)
        ax[0, t].scatter(np.arange(self.N), pred[t], marker='o', c='r', alpha=0.5)
        ax[0, t].set_xlabel('SNP')

        ax[1, t].scatter(pred[t], self.Y[t], marker='x', c='k', alpha=0.5)
        ax[0, t].set_title('Tissue: {}'.format(self.tissue_ids[t]))
        ax[1, t].set_xlabel('prediction')

    ax[0, 0].set_ylabel('observed')
    ax[1, 0].set_ylabel('observed')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_manhattan(self, component, thresh=0.0, save_path=None):
    """
    make manhattan plot for tissues, colored by lead snp of a components
    include tissues with p(component active in tissue) > thresh
    """
    logp = - norm.logcdf(-np.abs(self.Y)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])
    #sorted_tissues = np.flip(np.argsort(self.active[:, component]))
    #active_tissues = sorted_tissues[self.active[sorted_tissues, component] > thresh]
    active_tissues = np.arange(self.T)[self.active[:, component] > thresh]
    fig, ax = plt.subplots(1, active_tissues.size, figsize=(5*active_tissues.size, 4), sharey=True)
    for i, tissue in enumerate(active_tissues):
        lead_snp = self.pi[:, component].argmax()
        r2 = self.X[lead_snp]**2
        ax[i].scatter(pos, logp[tissue], c=r2, cmap='RdBu_r')
        ax[i].set_title('Tissue: {}\nLead SNP {}\nweight= {:.2f}, p={:.2f}'.format(
            self.tissue_ids[tissue], lead_snp, self.weights[tissue, component],self.active[tissue, component]))
        ax[i].set_xlabel('SNP')

    ax[0].set_ylabel('-log(p)')

    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_colocalizations(self, save_path=None, show=True):
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    sns.heatmap(self.get_A_intersect_B_coloc(), cmap='Blues', ax=ax[0], cbar=False, square=True)
    ax[0].set_title('At least one (intersect)')

    sns.heatmap(self.get_A_in_B_coloc(), cmap='Blues', ax=ax[1], yticklabels=False, cbar=False, square=True)
    ax[1].set_title('A in B (subset)')

    sns.heatmap(self.get_A_equals_B_coloc(), cmap='Blues', ax=ax[2], yticklabels=False, cbar=False, square=True)
    ax[2].set_title('A = B (all)')

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()

def plot_component_colocalizations(self, save_path=None, show=True):
    active_components = self.active.max(0) > 0.5

    component_colocalization = self.get_component_coloc()
    fig, ax = plt.subplots(1, active_components.sum(), figsize=(6 * active_components.sum(), 6))
    for i, k in enumerate(np.arange(self.K)[active_components]):
        if i == 0:
            yticklabels = self.tissue_ids
        else:
            yticklabels = []
        sns.heatmap(component_colocalization[k], ax=ax[i], square=True, cmap='Blues',
            yticklabels=yticklabels, xticklabels=[], cbar=False, vmin=0, vmax=1)
        ax[i].set_title('Component {}'.format(k))

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()
