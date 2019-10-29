import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os, sys, pickle

class SpikeSlabSER:
    def __init__(self, X, Y, K, snp_ids, tissue_ids, prior_activity, prior_variance, tolerance=1e-6, **kwargs):
        """
        X [N x N] covariance matrix
            if X is [T x N x N] use seperate embedding for each tissue
        Y [T x N] matrix, Nans should be converted to 0s?
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
        self.global_sign = np.ones(self.N)

        self.weights = np.random.random((self.T, self.K)) * 5
        #self.weights = np.ones((self.T, self.K))
        self.active = np.ones((self.T, self.K))
        #self.active = np.random.random((self.T, self.K))

        self.elbos = []
        self.tolerance = tolerance

    ################################
    # UPDATE AND FITTING FUNCTIONS #
    ################################
    def _flip(self, k, thresh=0.9):
        """
        flip snps and zscores to avoid having two blocks of negatively correlated snps
        k is the component to operate on
        thresh controls how far from the lead snp of the component to flip signs on

        the model keeps a global record of whats changed
        """
        sign = np.ones(self.N)
        lead_snp = self.pi[:, k].argmax()
        switch = (self.X[lead_snp] < 0) & (np.abs(self.X[lead_snp]) > thresh)
        sign[switch] *= -1

        # update data and ld matrices to relfect this change in sign
        self.Y = (self.Y * sign).astype(np.float64)
        if self.X.ndim == 2:
            self.X = self.X * np.outer(sign, sign).astype(np.float64)
        else:
            self.X = self.X * np.outer(sign, sign).astype(np.float64)[None]
        self.global_sign = self.global_sign * sign

    def _compute_residual(self, k=None):
        """
        residual computation, works when X is 2d or 3d
        k is a component to exclude from residual computation
        """
        prediction = self._compute_prediction(k)
        residual = self.Y - prediction
        return residual

    def _compute_prediction(self, k=None):
        W = self.weights * self.active
        if self.X.ndim == 2:
            prediction = W @ (self.X @ self.pi).T
            if k is not None:
                prediction -= W[:, k][:, None] * (self.X @ self.pi[:, k])[None]
        if self.X.ndim == 3:
            prediction = np.stack([W[t] @ (self.X[t] @ self.pi).T for t in range(self.T)])
            if k is not None:
                prediction -= np.stack(
                    [W[t, k] * (self.X[t] @ self.pi[:, k]) for t in range(self.T)]
                )
        return prediction

    def update_weights(self, components=None):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.K)

        old_weights = self.weights.copy()
        weight_var = 1 / (1 + (1 /self.prior_variance))

        for k in components:
            # get expected weights
            W = self.weights * self.active

            # compute residual
            #residual = self.Y - W @ (self.X @ self.pi).T
            # remove effect of kth component from residual
            #r_k = residual + (W[:, k])[:, None] * (self.X @ self.pi[:, k])[None]
            r_k = self._compute_residual(k)
            # update p(w | s = 1)
            self.weights[:, k] = (weight_var) * (r_k @ self.pi[:, k])

        weight_diff = np.abs(old_weights - self.weights).max()
        return weight_diff

    def update_active(self, components=None):
        """
        X is LD/Covariance Matrix
        Y is T x N
        weights  T x K matrix of weight parameters
        active T x K active[t, k] = logp(s_tk = 1)
        prior_activitiy
        """
        if components is None:
            components = np.arange(self.K)

        old_active = self.active.copy()
        weight_var = 1 / (1 + (1 /self.prior_variance))
        for k in components:
            # get expected weights
            #W = self.weights * self.active

            # compute residual
            #residual = self.Y - W @ (self.X @ self.pi).T

            # remove effect of kth component from residual [t, n]
            #r_k = residual + (W[:, k])[:, None] * (self.X @ self.pi[:, k])[None]
            r_k = self._compute_residual(k)
            for t in range(self.T):
                # q(s = 0)
                off =  -0.5 + np.log(1 - self.prior_activity[k]) + 0.5 * np.log(self.prior_variance)

                # q(s = 1)
                on = r_k[t] @ self.pi[:, k] * self.weights[t, k] - 0.5 * (self.weights[t, k]**2 + weight_var) \
                    - 0.5 * (1 / self.prior_variance) * (self.weights[t, k]**2 + weight_var) \
                    + np.log(self.prior_activity[k]) \
                    + 0.5 * np.log(weight_var)

                u = np.array([on, off])
                u = np.exp(u - u.max())
                u = u / u.sum()
                self.active[t, k] = u[0]

        active_diff = np.abs(old_active - self.active).max()
        return active_diff

    def update_pi(self, components=None):
        """
        update pi
        """
        if components is None:
            components = np.arange(self.K)

        old_pi = self.pi.copy()
        W = self.weights * self.active

        active_components = (self.active.max(0)[components] > 1e-2)[:self.K]

        for k in components[active_components]:
            # compute residual
            #residual = self.Y - W @ (self.X @ self.pi).T

            # remove effect of kth component from residual
            #r_k = residual + (W[:, k])[:, None] * (self.X @ self.pi[:, k])[None]
            r_k = self._compute_residual(k)

            # r_k^T @ Sigma_inv @ (Sigma @ pi) @ (weights * beta)
            pi_k = r_k * W[:, k][:, None]
            pi_k = pi_k.sum(0)

            # normalize to probabilities
            pi_k = np.exp(pi_k - pi_k.max() + 5)
            pi_k = pi_k / pi_k.sum()
            self.pi[:, k] = pi_k

        # component not active-- back to prior
        for k in components[~active_components]:
            self.pi[:, k] = np.ones(self.N) / self.N

        pi_diff = np.abs(self.pi - old_pi).max()
        return pi_diff

    def _fit(self, max_inner_iter=1, max_outer_iter=1000, bound=False, verbose=False, components=None, diffuse=1.0):
        """
        loop through updates until convergence
        """
        self.elbos.append(self.compute_elbo())
        for i in range(max_outer_iter):
            # update weights and activities
            for _ in range(max_inner_iter):
                diff1 = self.update_weights(components)
                diff2 = self.update_active(components)
                if diff1 < self.tolerance and diff2 < self.tolerance:
                    break
            self.elbos.append(self.compute_elbo())

            # update pi
            for _ in range(max_inner_iter):
                diff3 = self.update_pi()
                self._diffuse_pi(diffuse)
                if diff3 < self.tolerance:
                    break
            self.elbos.append(self.compute_elbo())
            if np.abs(self.elbos[-1] - self.elbos[-3]) < self.tolerance:
                if verbose:
                    print('Parameters converged at iter {}'.format(i))
                break
            if diff1 < self.tolerance and diff2 < self.tolerance and diff3 < self.tolerance:
                if verbose:
                    print('Parameters converged at iter {}'.format(i))
                break

    def _forward_fit_step(self, l, max_inner_iter=1, max_outer_iter=1000, diffuse=1.0, bound=False, verbose=False, restarts=1, plots=False, quantile=0.0):
        """
        fit self as though there were only l components
        T initializations with unit weight at each tissue, pick best solution among them
        """
        K = self.K
        init_pi = self.pi.copy()
        init_active = self.active.copy()
        init_weights = self.weights.copy()

        restart_dict = {}
        elbos = []

        residual = self._compute_residual()
        sq_err = np.max(residual**2, axis=0)
        pi = sq_err * (sq_err > np.quantile(sq_err, quantile))
        pi = pi / pi.sum()

        if plots:
            plt.scatter(np.arange(pi.size), (pi))
            # plt.show()

        # positive initialization
        for i in range(restarts):
            self.pi = init_pi
            self.active = init_active
            self.weights = init_weights

            # initialize activity to random
            active_t = init_active.copy()
            active_t[:, l-1] = np.random.random(self.T)

            # initialize weights something random
            t = np.random.choice(self.T)
            weights_t = init_weights.copy()
            #weights_t[:, l-1] = np.random.normal(size=self.T)
            #weights_t[:, l-1] = np.eye(self.T)[t]
            weights_t[:, l-1] = np.zeros_like(self.T)

            # initialize pi of the lth component, weigh snps with poor predictions heavier
            pi_t = init_pi.copy()
            pi_t[:, l-1] = pi

            self.active = active_t
            self.weights = weights_t
            self.pi = pi_t

            self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l-1, l), diffuse=diffuse)
            self._flip(k=l-1, thresh=0.9)
            self._fit(max_inner_iter, max_outer_iter, bound, verbose, components=np.arange(l), diffuse=diffuse)

            restart_dict[i] = (self.pi.copy(), self.active.copy(), self.weights.copy())
            elbos.append(self.compute_elbo())

        select = np.argmax(elbos)
        self.elbos.append(elbos[select])
        new_pi, new_active, new_weights = restart_dict[select]

        self.pi = new_pi
        self.active = new_active
        self.weights = new_weights

        return restart_dict, elbos

    def forward_fit(self, early_stop=False, max_inner_iter=1, max_outer_iter=1000, diffuse=1.0, bound=False, verbose=False, restarts=1, plots=False, quantile=0.0):
        """
        forward selection scheme for variational optimization
        fit first l components with weights initialized to look at each tissue
        select the best solution (by elbo) among tissue initialization

        fit first l+1 components
        """
        self.weights = np.zeros_like(self.weights)
        self.active = np.ones_like(self.active)

        for l in range(1, self.K+1):
            print('Forward fit, learning {} components'.format(l))
            self._forward_fit_step(
                l, max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter,
                bound=bound, verbose=verbose, restarts=restarts,
                diffuse=diffuse, plots=plots, quantile=quantile)

            if plots:
                self.plot_components()

            # if the next step turned off the component, all future steps will
            # zero them out and do a final fit of the self
            # if self.pi[:, l-1].max() < 0.01 and early_stop:
            if early_stop and self.active[:, l-1].max() < 0.5:
                print('learned inactive cluster, finalizing parameters')
                # zero initialize the components
                self.active[:, l:] = 1 - self.prior_activity[l:]
                self.weights[:, l:] = 0
                self._fit(max_inner_iter=max_inner_iter, max_outer_iter=max_outer_iter, bound=bound, verbose=verbose, diffuse=diffuse)
                if plots:
                    self.plot_components()
                break

    def diffusion_fit(self, schedule):
        for i, rate in enumerate(schedule):
            self._fit(max_outer_iter=5, verbose=True)
            transition = np.abs(self.X) * (np.abs(self.X) > rate)
            degree = np.diag(1 / (transition.sum(1)))
            transition = degree @ transition
            self.pi = transition.T @ self.pi

    def _diffuse_pi(self, width):
        if width < 1.0:
            X = np.abs(self.X)
            if X.ndim == 3:
                X = np.mean(X, axis=0)
            transition = X * (X >= width)
            degree = np.diag(1 / (transition.sum(1)))
            transition = degree @ transition
            self.pi = transition.T @ self.pi

    def compute_elbo(self):
        bound = 0 
        W = self.weights * self.active
        weight_var = self.prior_variance / (1 + self.prior_variance)

        if self.X.ndim == 2:
            Kzz = self.pi.T @ self.X @ self.pi
            Kzz = Kzz + np.diag(np.ones(self.K) - np.diag(Kzz))
            self.X
        for t in range(self.T):
            if self.X.ndim == 3:
                Kzz = self.pi.T @ self.X[t] @ self.pi
                Kzz = Kzz + np.diag(np.ones(self.K) - np.diag(Kzz))
            bound += self.Y[t] @ (self.pi @ W[t])
            bound += -0.5 * W[t] @ Kzz @ W[t]
            #bound += -0.5 * np.sum(
            #    self.active[t] * weight_var
            #    + (1 - self.active[t]) * self.prior_variance
            #    + (self.active[t] - self.active[t]**2) * self.weights[t]**2
            #)
            bound += -0.5 * np.sum(
                (self.weights[t]**2 + weight_var) * self.active[t] - (self.weights[t] * self.active[t])**2
            )

        KL = 0
        for t in range(self.T):
            for k in range(self.K):

                # KL (q(w|s) || p(w | s))
                KL += normal_kl(self.weights[t, k], weight_var, 0, self.prior_variance) * self.active[t, k]
                KL += normal_kl(0, self.prior_variance, 0, self.prior_variance) * (1 - self.active[t, k])

                # KL (q(s) || p(s))
                KL += categorical_kl(
                    np.array([self.active[t, k], 1 - self.active[t, k]]),
                    np.array([self.prior_activity[k], 1 - self.prior_activity[k]])
                )

        for k in range(self.K):
            # KL (q(z) || p (z))
            KL += categorical_kl(self.pi[:, k], np.ones(self.N) / self.N)
        bound -= KL
        return bound 

    def sort_components(self):
        """
        reorder components so that components with largest weights come first
        """
        order = np.flip(np.argsort(np.abs(self.weights).max(0)))
        self.weights = self.weights[:, order]
        self.active = self.active[:, order]
        self.pi = self.pi[:, order]

    def save(self, output_dir, model_name):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        pickle.dump(self.__dict__, open('{}/{}'.format(output_dir, model_name), 'wb'))

    #########################
    # MODEL QUERY FUNCTIONS #
    #########################

    def get_top_snp_per_component(self):
        """
        returns snp with highest posterior probability for each component
        returns p(max_snp) for each components
        """
        return self.snp_ids[self.pi.argmax(axis=0)], self.pi.max(axis=0)

    def get_credible_sets(self, alpha=0.9, thresh=0.5):
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

    def get_pip(self):
        """
        return posterior inclusion probability for each tissue, SNP pair
        PIP is the probability that a SNP is the causal snp in at least one component
        """
        pip = np.zeros((self.N, self.T))
        for t in range(self.T):
            for n in range(self.N):
                pip[n, t] = 1 - np.exp(np.sum([np.log(1 - self.pi[n, k] * self.active[t, k]) for k in range(self.K)]))

        return pd.DataFrame(pip, index=self.snp_ids, columns=self.tissue_ids)

    def get_A_in_B_coloc(self):
        """
        returns pairwise probability that active components in tissue A
        are a subset of the active components of tissue B

        ie component k on in A -> k on in B
           component k off in B -> k off in A
        """
        probs = np.zeros((self.T, self.T))
        for t1 in range(self.T):
            for t2 in range(self.T):
                A = self.active[t1]
                B = self.active[t2]
                # 1 - p(component OFF in t2 and component ON in t1)
                probs[t1, t2] = np.exp(np.sum(np.log((1 - (1-B) * A) + 1e-10)))
            probs += np.eye(self.T) - np.diag(np.diag(probs))
            
        return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)

    def get_A_equals_B_coloc(self):
        """
        return pairwise probability that the active components in two tissues are identicle
        component k on in A <--> k on in B
        """
        probs = np.zeros((self.T, self.T))
        for t1 in range(self.T):
            for t2 in range(self.T):
                A = self.active[t1]
                B = self.active[t2]
                # prod(p(component on in both tissues) +p(component off in both tissues))
                probs[t1, t2] = np.exp(np.sum(np.log(A * B + (1- A) * (1-B) + 1e-10)))
            probs += np.eye(self.T) - np.diag(np.diag(probs))
            
        return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)

    def get_A_intersect_B_coloc(self):
        """
        return pairwise probability that at least one component is shared by both tissues
        component k on in A AND k on in B for some k
        """
        probs = np.zeros((self.T, self.T))
        for t1 in range(self.T):
            for t2 in range(self.T):
                A = self.active[t1]
                B = self.active[t2]
                # 1 - p(all components off in both tissues)
                probs[t1, t2] = 1 - np.exp(np.sum(np.log((1 - A * B) + 1e-10)))


            probs += np.eye(self.T) - np.diag(np.diag(probs))
        return pd.DataFrame(probs, index=self.tissue_ids, columns=self.tissue_ids)

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

        sns.heatmap(self.weights[:, active_components], annot=False, cmap='RdBu_r', ax=ax[1], yticklabels=[])
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
                    ax[1, i].scatter(pos, predk, marker='o', alpha=0.5, label='k{}'.format(k))
                else:
                    ax[1, i].scatter(pos, predk, marker='o', alpha=0.5)
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

    def plot_colocalizations(self, save_path=None):
        fig, ax = plt.subplots(1, 3, figsize=(20, 8))
        sns.heatmap(self.get_A_intersect_B_coloc(), cmap='Blues', ax=ax[0], cbar=False, square=True)
        ax[0].set_title('At least one (intersect)')

        sns.heatmap(self.get_A_in_B_coloc(), cmap='Blues', ax=ax[1], yticklabels=False, cbar=False, square=True)
        ax[1].set_title('A in B (subset)')

        sns.heatmap(self.get_A_equals_B_coloc(), cmap='Blues', ax=ax[2], yticklabels=False, cbar=False, square=True)
        ax[2].set_title('A = B (all)')

        if save_path is not None:
            plt.savefig(save_path)
        # plt.show()
        plt.close()

def unit_normal_kl(mu_q, var_q):
    """
    KL (N(mu, var) || N(0, 1))
    """
    KL = 0.5 * (var_q + mu_q ** 2 - np.log(var_q) - 1)
    return KL

def normal_kl(mu_q, var_q, mu_p, var_p):
    KL = 0.5 * (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + 2 * np.log(np.sqrt(var_p) / np.sqrt(var_q)))
    return KL

def categorical_kl(pi_q, pi_p):
    """
    KL(pi_q || pi_p)
    """
    return np.sum(pi_q * (np.log(pi_q + 1e-10) - np.log(pi_p + 1e-10)))

