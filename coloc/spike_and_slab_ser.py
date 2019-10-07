import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class SpikeSlabSER:
    def __init__(self, X, Y, K, snp_ids, tissue_ids, prior_activity, prior_variance):
        """
        X N x N covariance matrix
        """

        self.X = X
        self.Y = Y
        self.K = K
        self.T = Y.shape[0]
        self.N = Y.shape[1]

        self.prior_activity = prior_activity
        self.prior_variance = prior_variance

        self.snp_ids = snp_ids
        self.tissue_ids = tissue_ids

        # initialize variational parameters
        pi = np.random.random((self.N, self.K)) + 1
        pi = pi / pi.sum(0)

        self.pi = pi
        self.weights = np.random.random((self.T, self.K)) * 5
        #self.weights = np.ones((self.T, self.K))
        self.active = np.ones((self.T, self.K))
        #self.active = np.random.random((self.T, self.K))
        self.elbos = []


    def update_ss_weights(self):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        old_weights = self.weights.copy()
        old_active = self.active.copy()

        weight_var = self.prior_variance / (1 + self.prior_variance)
        for k in range(self.K):
            # get expected weights
            W = self.weights * self.active

            # compute residual
            residual = self.Y - W[:, :self.K] @ (self.X @ self.pi[:, :self.K]).T

            # remove effect of kth component from residual
            r_k = residual + (W[:, k])[:, None] * (self.X @ self.pi[:, k])[None]

            # update p(w | s = 1)
            self.weights[:, k] = (weight_var) * r_k @ self.pi[:, k]

            # now update p(s = 1)
            on = r_k @ self.pi[:, k] * self.weights[:, k] \
                - 0.5 * (self.weights[:, k]**2 + weight_var) + np.log(self.prior_activity[k])
            # on = on - on.max()

            normalizer = np.log(np.exp(on) + (1-self.prior_activity[k]))
            self.active[:, k] = np.exp(on - normalizer)

        weight_diff = np.abs(old_weights - self.weights).max()
        active_diff = np.abs(old_active - self.active).max()
        return np.array([weight_diff, active_diff]).max()

    def update_pi(self):
        """
        update pi
        """
        old_pi = self.pi.copy()

        W = self.weights * self.active
        active_components = (self.active.max(0) > 1e-2)[:self.K]

        for k in np.arange(self.K)[active_components]:
            # compute residual
            residual = self.Y - W[:, :self.K] @ (self.X @ self.pi[:, :self.K]).T

            # remove effect of kth component from residual
            r_k = residual + (W[:, k])[:, None] * (self.X @ self.pi[:, k])[None]

            # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
            pi_k = r_k * W[:, k][:, None]
            pi_k = pi_k.sum(0)

            # normalize to probabilities
            pi_k = np.exp(pi_k - pi_k.max() + 10)
            pi_k = pi_k / pi_k.sum()

            # if nans dont update
            # this can happen if pi_k is 0 or constant
            # should revert to prior but that might be disadvantageous
            #if ~np.any(np.isnan(pi_k)):
            self.pi[:, k] = pi_k

        # component not active-- back to prior
        for k in np.arange(self.K)[~active_components]:
            self.pi[:, k] = np.ones(self.N) / self.N

        pi_diff = np.abs(self.pi - old_pi).max()
        return pi_diff

    def fit(self, max_inner_iter=100, max_outer_iter=1000, bound=False, verbose=False):
        """
        loop through updates until convergence
        """
        for i in range(max_outer_iter):
            for j in range(max_inner_iter):
                inner_diff = self.update_pi()
                if inner_diff < 1e-8:
                    break

                if j % 10 == 0 and bound:
                    self.elbos.append(self.compute_elbo())

            diff = self.update_ss_weights()
            if bound:
                self.elbos.append(self.compute_elbo())

            if diff < 1e-8:
                for _ in range(max_inner_iter):
                    diff = self.update_pi()
                    if diff < 1e-8:
                        break
                if verbose:
                    print('Parameters converged at iter {}'.format(i))
                break

    def fit_l(self, l=1, max_inner_iter=100, max_outer_iter=1000, bound=False, verbose=False):
        """
        fit self with l+1, ... K components set to 0
        loop through updates until convergence

        fit as though there were
        """
        K = self.K
        self.K = l
        self.fit(max_inner_iter, max_outer_iter, bound, verbose)
        self.K = K

    def _forward_fit_step(self, l, max_inner_iter=100, max_outer_iter=1000, bound=False, verbose=False):
        """
        fit self as though there were only l components
        T initializations with unit weight at each tissue, pick best solution among them
        """
        init_pi = self.pi.copy()
        init_active = self.active.copy()
        init_weights = self.weights.copy()

        restart_dict = {}
        elbos = []
        for t in range(self.T):
            active_t = init_active.copy()
            active_t[:, l-1] = np.eye(self.T)[t]

            weights_t = init_weights.copy()
            weights_t[:, l-1] = np.eye(self.T)[t]

            self.active = active_t.copy()
            self.weights = init_weights.copy()

            self.pi = init_pi.copy()

            self.fit_l(l, max_inner_iter, max_outer_iter, bound, verbose)
            restart_dict[t] = (self.pi.copy(), self.active.copy(), self.weights.copy())
            elbos.append(self.compute_elbo())

        select = np.argmax(elbos)
        self.elbos.append(elbos[select])
        new_pi, new_active, new_weights = restart_dict[select]

        self.pi = new_pi
        self.active = new_active
        self.weights = new_weights

        return restart_dict, elbos

    def forward_fit(self, early_stop=False, max_inner_iter=100, max_outer_iter=1000, bound=False, verbose=False):
        """
        forward selection scheme for variational optimization
        fit first l components with weights initialized to look at each tissue
        select the best solution (by elbo) among tissue initialization

        fit first l+1 components
        """
        for l in range(1, self.K+1):
            print('Forward fit, learning {} components'.format(l))
            self._forward_fit_step(l, max_inner_iter, max_outer_iter, bound, verbose)

            # if the next step turned off the component, all future steps will
            # zero them out and do a final fit of the self
            if self.pi[:, l-1].max() < 0.01 and early_stop:
                print('learned inactive cluster, finalizing parameters')
                self.active[:, l:] = 0
                self.fit()
                break

    def compute_elbo(self):
        Kzz = self.pi.T @ self.X @ self.pi
        Kzz = Kzz + np.diag(np.ones(self.K) - np.diag(Kzz))
        W = self.weights * self.active

        bound = 0
        for t in range(self.T):
            bound += self.Y[t] @ (self.pi @ W[t])
            bound -= 0.5 * W[t] @ (Kzz @ W[t])

        weight_var = self.prior_variance / (1 + self.prior_variance)
        varW = self.active  * (weight_var) + (1 - self.active) * self.prior_variance
        bound -= 0.5 * varW.sum()

        KL = unit_normal_kl(self.weights, weight_var).sum()
        KL += np.sum([categorical_kl(pi, np.ones(self.N)/self.N) for pi in self.pi.T])
        KL += np.sum([[categorical_kl(
            np.array([self.active[t, k], 1-self.active[t, k]]),
            np.array([self.prior_activity[k], 1-self.prior_activity[k]])) for k in range(self.K)] for t in range(self.T)])
        bound -= KL
        return bound 

    def get_top_snp_per_component(self):
        """
        returns snp with highest posterior probability for each component
        returns p(max_snp) for each components
        """
        return self.snp_ids[self.pi.argmax(axis=0)], self.pi.max(axis=0)

    def get_confidence_sets(self, alpha=0.9, thresh=0.1):
        """
        return snps for active components
        """
        confidence_sets = {}
        active = self.active.max(0) > thresh
        for k in np.arange(self.K)[active]:
            cset_size = (np.cumsum(np.flip(np.sort(self.pi[:, k]))) < alpha).sum() + 1
            cset = np.flip(np.argsort(self.pi[:, k])[-cset_size:])
            confidence_sets[k] = self.snp_ids[cset]
        return confidence_sets

    def plot_components(self, thresh=0.1):
        """
        plot inducing point posteriors, weight means, and probabilities
        """
        W = self.weights * self.active
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

        sns.heatmap(self.weights[:, active_components], annot=False, cmap='RdBu_r', ax=ax[1])
        ax[1].set_title('weights')
        ax[1].set_xlabel('component')

        sns.heatmap((self.active)[:, active_components],
            annot=False, cmap='Blues', ax=ax[0],
            vmin=0, vmax=1, yticklabels=self.tissue_ids)
        ax[0].set_title('active')
        ax[0].set_xlabel('component')

        plt.show()
        plt.close()

    def plot_predictions(self):
        """
        plot predictions against observed z scores
        """
        pred = (self.weights * self.active) @ (self.X @ self.pi).T
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
        plt.show()
        plt.close()

    def plot_manhattan(self, component, thresh=0.0):
        """
        make manhattan plot for tissues, colored by lead snp of a components
        include tissues with p(component active in tissue) > thresh
        """
        logp = -np.log(norm.cdf(-np.abs(self.Y)) * 2)

        sorted_tissues = np.flip(np.argsort(self.active[:, component]))
        active_tissues = sorted_tissues[self.active[sorted_tissues, component] > thresh]
        fig, ax = plt.subplots(1, active_tissues.size, figsize=(5*active_tissues.size, 4), sharey=True)
        for i, tissue in enumerate(active_tissues):
            lead_snp = self.pi[:, component].argmax()
            r2 = self.X[lead_snp]**2
            ax[i].scatter(np.arange(self.N), logp[tissue], c=r2, cmap='RdBu_r')
            ax[i].set_title('Tissue: {}\nLead SNP {}\nweight= {:.2f}, p={:.2f}'.format(self.tissue_ids[tissue], lead_snp, self.weights[tissue, component], self.active[tissue, component]))
            ax[i].set_xlabel('SNP')

        ax[0].set_ylabel('-log(p)')

    def get_component_colocalization(self, component):
        """
        return probability of colocalization in a component
        """
        df = pd.DataFrame(self.active[:, component] * self.active[:, component][:, None], index=self.tissue_ids, columns=self.tissue_ids)
        return df

    def get_global_colocalzation(self):
        """
        return pandas data frame with pairwise probability of colocalzation in at least one component
        """
        failure = np.ones((self.T, self.T))
        for k in range(self.K):
            failure *= (1 - self.active[:, k] * self.active[:, k][:, None])
        df = pd.DataFrame(1 - failure, index=self.tissue_ids, columns=self.tissue_ids)
        return df

def update_ss_weights(X, Y, weights, active, pi, prior_activity, prior_variance=1.0):
    """
    X is LD/Covariance Matrix
    Y is T x N
    weights  T x K matrix of weight parameters
    active T x K active[t, k] = logp(s_tk = 1)
    prior_activitiy
    """
    old_weights = weights.copy()
    old_active = active.copy()

    T, K = weights.shape
    W = weights * np.exp(active)
    weight_var = prior_variance / (1 + prior_variance)
    for k in range(K):
        # get expected weights
        W = weights * np.exp(active)

        # compute residual
        residual = Y - (W) @ (X @ pi).T

        # remove effect of kth component from residual
        r_k = residual + (W[:, k])[:, None] * (X @ pi[:, k])[None]

        # update p(w | s = 1)
        weights[:, k] = (weight_var) * r_k @ pi[:, k]

        # now update p(s = 1)
        on = r_k @ pi[:, k] * weights[:, k] \
            - 0.5 * (weights[:, k]**2 + weight_var) + np.log(prior_activity[k])
        # on = on - on.max()

        normalizer = np.log(np.exp(on) + (1-prior_activity[k]))
        active[:, k] = (on - normalizer)

    weight_diff = np.abs(old_weights - weights).max()
    active_diff = np.abs(np.exp(old_active) - np.exp(active)).max()
    return np.array([weight_diff, active_diff]).max()


def update_pi(X, Y, weights, active, pi):
    old_pi = pi.copy()
    T, N = Y.shape
    K = pi.shape[1]

    W = weights * np.exp(active)
    active_components = np.exp(active).max(0) > 1e-2

    for k in np.arange(K)[active_components]:
        # compute residual
        residual = Y - W @ (X @ pi).T

        # remove effect of kth component from residual
        r_k = residual + (W[:, k])[:, None] * (X @ pi[:, k])[None]

        # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
        pi_k = r_k * W[:, k][:, None]
        pi_k = pi_k.sum(0)

        # normalize to probabilities
        pi_k = np.exp(pi_k - pi_k.max() + 10)
        pi_k = pi_k / pi_k.sum()

        # if nans dont update
        # this can happen if pi_k is 0 or constant
        # should revert to prior but that might be disadvantageous
        #if ~np.any(np.isnan(pi_k)):
        pi[:, k] = pi_k

    for k in np.arange(K)[~active_components]:
        pi[:, k] = np.ones(N)/N

    pi_diff = np.abs(pi - old_pi).max()
    return pi_diff


def unit_normal_kl(mu_q, var_q):
    """
    KL (N(mu, var) || N(0, 1))
    """
    KL = 0.5 * (var_q + mu_q ** 2 - np.log(var_q) - 1)
    return KL


def categorical_kl(pi_q, pi_p):
    """
    KL(pi_q || pi_p)
    """
    return np.sum(pi_q * (np.log(pi_q + 1e-10) - np.log(pi_p + 1e-10)))

