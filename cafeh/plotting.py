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

def plot_component_correlations(self, save_path=None):
    sns.heatmap(np.abs(np.corrcoef((self.X @ self.pi.T).T)), cmap='Reds')
    plt.title('Component correlation')
    plt.xlabel('Component')
    plt.ylabel('Component')
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_component_x_component(self, save_path=None, show=True):
    active = np.any(self.active > 0.5, axis=0)
    components = (self.X @ self.pi.T)[:, active]
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

def plot_pips(self, thresh=0.5, star=None, save_path=None, show=True):
    pips = self.get_pip()
    cs, purities = self.get_credible_sets()
    plt.scatter(np.arange(self.dims['N']), pips, c='k')

    for k in cs.keys():
        if purities[k] > thresh:
            plt.scatter(np.arange(self.dims['N'])[cs[k]], pips[cs[k]], marker='o', s=200, alpha=0.5)
    #plt.scatter(np.arange(self.dims['N'])[causal1], pips[causal1], c='r', marker='*')
    #plt.scatter(np.arange(self.dims['N'])[causal2], pips[causal2], c='y', marker='*')


    plt.xlabel('SNP')
    plt.ylabel('PIP')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_credible_sets_ld(self, snps=None, alpha=0.9, thresh=0.5, save_path=None, show=True):
    if snps is None:
        snps = []
    credible_sets, purity = self.get_credible_sets(alpha=alpha)
    for k in np.arange(self.dims['K']):
        if purity[k] > thresh:
            cset = np.arange(self.dims['N'])[np.isin(self.snp_ids, credible_sets[k])]
            snps.append(cset)

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

def plot_credible_sets_r2(self, snps=None, alpha=0.9, thresh=0.5, save_path=None, show=True):
    if snps is None:
        snps = []
    credible_sets, purity = self.get_credible_sets(alpha=alpha)
    for k in np.arange(self.dims['K']):
        if purity[k] > thresh:
            cset = np.arange(self.dims['N'])[np.isin(self.snp_ids, credible_sets[k])]
            snps.append(cset)

    sizes = np.array([x.size for x in snps])
    snps = np.concatenate(snps)
    
    fig, ax = plt.subplots(1, figsize=(6, 5))
    ld = self.get_ld(snps=snps)
    if np.ndim(ld) == 2:
        r2 = ld ** 2
    else:
        r2 = (ld ** 2).min(0)
    sns.heatmap(
        r2, cmap='RdBu_r', vmin=0, vmax=1, ax=ax,
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

def plot_decomposed_manhattan(self, studys=None, components=None, save_path=None, show=True):
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]

    if components is None:
        components = np.arange(self.dims['K'])[np.any((self.active > 0.5), 0)]

    W = self.active * self.weight_means
    c = (self.X @ self.pi.T)

    pred = self.compute_prediction()
    logp = -norm.logcdf(-np.abs(pred)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    fig, ax = plt.subplots(2, studys.size, figsize=((studys.size)*4, 6), sharey=False)
    for i, t in enumerate(studys):
        ulim = []
        llim = []
        ax[0, i].set_title('{}\n-log p'.format(self.study_ids[t]))
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
            predk = self.compute_prediction() - self._compute_prediction(k=k)
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

def plot_decomposed_manhattan2(self, studys=None, width=None, components=None, save_path=None):
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]

    if components is None:
        components = np.arange(self.dims['K'])[np.any((self.active > 0.5), 0)]

    if width is None:
        width = int(np.sqrt(studys.size)) + 1
        height = width
    else:
        height = int(studys.size / width) + 1

    pred = ((self.active * self.weight_means) @ (self.X @ self.pi.T).T)
    logp = - norm.logcdf(-np.abs(pred)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    W = self.active * self.weight_means
    c = (self.X @ self.pi.T)

    pred = self.compute_prediction()
    fig, ax = plt.subplots(height, width, figsize=(width*4, height*3), sharey=False)

    ax = np.array(ax).flatten()
    for i, t in enumerate(studys):
        ax[i].set_title('{}\nby component'.format(self.study_ids[t]))

        for k in components:
            predk = self.compute_prediction() - self._compute_prediction(k=k)
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

def plot_decomposed_zscores(self, studys=None, components=None, thresh=0.9, save_path=None, show=True):
    """
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]
    """
    studys = np.arange(self.dims['T'])
    cs, p = self.get_credible_sets()
    components = np.array([k for k in range(self.dims['K']) if p[k] > 0.5])

    pred = self.compute_prediction()
    pos = np.arange(self.snp_ids.size)

    fig, ax = plt.subplots(2, studys.size, figsize=((studys.size)*4, 6), sharey=False)
    ax = np.atleast_2d(ax)
    if ax.shape[0] == 1:
        ax = ax.T

    for t in studys:
        ax[0, t].set_title('{}\nzscores'.format(self.study_ids[t]))
        ax[1, t].set_title('components')
        ax[1, t].set_xlabel('SNP position')

        if t == 0:
            ax[0, t].scatter(pos, self.Y[t], marker='x', c='k', alpha=0.5, label='zscore')
            ax[0, t].scatter(pos, pred[t], marker='o', c='r', alpha=0.5, label='prediction')
        else:
            ax[0, t].scatter(pos, self.Y[t], marker='x', c='k', alpha=0.5)
            ax[0, t].scatter(pos, pred[t], marker='o', c='r', alpha=0.5)

        if components.size > 0:
            for k in components:
                predk = self.compute_prediction_component(k)
                predkt = predk[t]
                #ax[1, t].scatter(pos, predkt, marker='o', alpha=self.active[t, k], label='k{}'.format(k))
                ax[1, t].scatter(pos, predkt, marker='o', alpha=self.active[t, k])
                #fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=(components.size)+2, borderaxespad=0.)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_decomposed_zscores2(self, studys=None, width=None, components=None, save_path=None):
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]

    if components is None:
        components = np.arange(self.dims['K'])[np.any((self.active > 0.5), 0)]

    if width is None:
        width = int(np.sqrt(studys.size)) + 1
        height = width
    else:
        height = int(studys.size / width) + 1

    pred = self.compute_prediction()
    logp = -np.log(norm.cdf(-np.abs(pred))*2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    W = self.active * self.weight_means
    c = (self.X @ self.pi.T)

    pred = W @ c.T
    fig, ax = plt.subplots(height, width, figsize=(width*4, height*3), sharey=False)

    ax = np.array(ax).flatten()
    for i, t in enumerate(studys):
        ax[i].set_title('{}\nby component'.format(self.study_ids[t]))

        for k in components:
            predk = self.compute_prediction() - self._compute_prediction(k=k)
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

def plot_residual_zscores(self, studys=None, components=None, save_path=None):
    """
    plot residual of study t with components removed
    """
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]

    if components is None:
        components = np.arange(self.dims['K'])[np.any((self.active > 0.5), 0)]

    W = self.active * self.weight_means
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])

    fig, ax = plt.subplots(components.size, studys.size, figsize=(studys.size*4, components.size*3), sharey=False)
    residual = self._compute_residual()
    for j, k in enumerate(components):
        for i, t in enumerate(studys):
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

            ax[j, i].set_title('{}\n{} removed'.format(self.study_ids[t], k))
            ax[j, i].set_xlabel('observed z score')
            ax[j, i].set_ylabel('residual z score')
            ax[j, i].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_residual_manhattan(self, studys=None, components=None, save_path=None):
    """
    plot residual of study t with components removed
    """
    if studys is None:
        studys = np.arange(self.dims['T'])
    else:
        studys = np.arange(self.dims['T'])[np.isin(self.study_ids, studys)]

    if components is None:
        components = np.arange(self.dims['K'])[np.any((self.active > 0.5), 0)]

    W = self.active * self.weight_means
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])
    logp = -norm.logcdf(-np.abs(self.Y)) - np.log(2)

    fig, ax = plt.subplots(components.size, studys.size, figsize=(studys.size*4, components.size*3), sharey=False)

    residual = self._compute_residual()
    residual_logp = -norm.logcdf(-np.abs(residual)) - np.log(2)

    for j, k in enumerate(components):
        for i, t in enumerate(studys):
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

            ax[j, i].set_title('{}\n{} removed'.format(self.study_ids[t], k))
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
    pred = self.compute_prediction()
    fig, ax = plt.subplots(2, self.dims['T'], figsize=(4*self.dims['T'], 6), sharey=True)
    for t in range(self.dims['T']):
        ax[0, t].scatter(np.arange(self.dims['N']), self.Y[t], marker='x', c='k', alpha=0.5)
        ax[0, t].scatter(np.arange(self.dims['N']), pred[t], marker='o', c='r', alpha=0.5)
        ax[0, t].set_xlabel('SNP')

        ax[1, t].scatter(pred[t], self.Y[t], marker='x', c='k', alpha=0.5)
        ax[0, t].set_title('study: {}'.format(self.study_ids[t]))
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
    make manhattan plot for studys, colored by lead snp of a components
    include studys with p(component active in study) > thresh
    """
    logp = - norm.logcdf(-np.abs(self.Y)) - np.log(2)
    pos = np.array([int(x.split('_')[1]) for x in self.snp_ids])
    #sorted_studys = np.flip(np.argsort(self.active[:, component]))
    #active_studys = sorted_studys[self.active[sorted_studys, component] > thresh]
    active_studys = np.arange(self.dims['T'])[self.active[:, component] > thresh]
    fig, ax = plt.subplots(1, active_studys.size, figsize=(5*active_studys.size, 4), sharey=True)
    for i, study in enumerate(active_studys):
        lead_snp = self.pi.T[:, component].argmax()
        r2 = self.X[lead_snp]**2
        ax[i].scatter(pos, logp[study], c=r2, cmap='RdBu_r')
        ax[i].set_title('study: {}\nLead SNP {}\nweight= {:.2f}, p={:.2f}'.format(
            self.study_ids[study], lead_snp, self.weight_means[study, component],self.active[study, component]))
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
    for i, k in enumerate(np.arange(self.dims['K'])[active_components]):
        if i == 0:
            yticklabels = self.study_ids
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
