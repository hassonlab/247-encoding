# Takes 90s to make one plot with 161 lags and 5000 perms of sig

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats import multitest

nperms = 5000
pdir = '/scratch/gpfs/zzada/247-encoding/results/podcast/'
dirs = ['0shot-zz-podcast-full-777-gpt2-xl/777/',
        '0shot-zz-podcast-full-777-gpt2-xl-nn/777/',
        '0shot-zz-podcast-full-777-gpt2-xl-shuffle/777/']
colors = ['blue', 'red', 'gray']
calcsig = True
experiments = pd.read_csv('data/experiments.csv')


def correlate(A, B, axis=0):
    """Calculate pearson correlation between two matricies.

       axis = 0 correlates columns in A to columns in B
       axis = 1 correlates rows in A to rows in B
    """
    assert A.ndim == B.ndim, 'Matrices must have same number of dimensions'
    assert A.shape == B.shape, 'Matrices must have same shape'

    A_mean = A.mean(axis=axis, keepdims=True)
    B_mean = B.mean(axis=axis, keepdims=True)
    A_stddev = np.sum((A - A_mean)**2, axis=axis)
    B_stddev = np.sum((B - B_mean)**2, axis=axis)

    num = np.sum((A - A_mean) * (B - B_mean), axis=axis)
    den = np.sqrt(A_stddev * B_stddev)

    return num / den


def one_samp_perm(x, nperms):
    n = len(x)
    dist = np.zeros(nperms)
    for i in range(nperms):
        dist[i] = np.random.choice(x, n, replace=True).mean()

    # s = np.sort(dist)  # unnecessary
    # val = np.sum(s > 0)
    val = np.sum(dist > 0)
    p_value = 1 - val / nperms
    # == np.mean(s < 0)
    return p_value


def paired_permutation(x, y, nperms):
    # Order of x and y matters
    n = len(x)
    truescore = (x - y).mean()
    dist = np.zeros(nperms)
    for i in range(nperms):
        s = np.random.choice([1, -1], n)
        dist[i] = np.mean(s * (x-y))

    p_value = (truescore > dist).mean()
    return p_value


def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(pvals,
                                        method='fdr_bh',
                                        is_sorted=False)
    return pcor


for experiment in experiments.columns:
    print(experiment)
    elecs = experiments.loc[:, experiment]

    fig, ax = plt.subplots()
    ax.axvline(0, ls='-', c='black', alpha=0.1)
    ax.axhline(0, ls='-', c='black', alpha=0.1)
    ax.set(xlabel='Lag (ms)', ylabel='Correlation (r+se)')

    allcorrs = []

    for i, resultdir in enumerate(dirs):
        lags = []
        signal = []
        pred_signal = []

        for elec in elecs:
            if pd.isna(elec):
                continue
            filename = pdir + resultdir + elec[2:5] + '_' + elec[30:] + '.pkl'
            if not os.path.isfile(filename):
                print(elec, 'is not found')
                continue
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                lags = data['lags']
                signal.append(data['Y_signal'])
                pred_signal.append(data['Yhat_signal'])

        signal = np.stack(signal, axis=-1)
        pred_signal = np.stack(pred_signal, axis=-1)

        corrs = []
        sems = []
        rawcorr = []
        for lag in range(signal.shape[1]):
            A = signal[:,lag,:]
            B = pred_signal[:,lag,:]
            rs = correlate(A, B, axis=1)
            corrs.append(rs.mean())
            sems.append(rs.std() / np.sqrt(len(rs)))
            rawcorr.append(rs)

        mean = np.asarray(corrs)
        err = np.asarray(sems)
        nelecs = signal.shape[-1]
        lags = np.asarray(lags)

        ax.plot(lags, corrs, color=colors[i])
        ax.fill_between(lags, mean - err, mean + err, alpha=0.1, color=colors[i])
        ax.set_title(f'{experiment} | N={nelecs}')

        allcorr = np.vstack(rawcorr)
        allcorrs.append(allcorr)

    if calcsig:
        corrs = allcorrs[0]
        pvals = [one_samp_perm(corrs[i], nperms) for i in range(len(lags))]
        pcorr = fdr(pvals)
        m = corrs.mean(axis=1)
        g = pcorr <= 0.01
        if g.any():
            sigSig = m[g]
            if (pcorr > 0.01).any():
                maxP = m[pcorr > 0.01].max()
                gg = sigSig > maxP
                if gg.any():
                    minP = sigSig[gg].min()
                    ax.axhline(minP)

        corrs2 = allcorrs[1]
        pvals = [paired_permutation(corrs2[i], corrs[i], nperms) for i in range(len(lags))]
        pcorr = fdr(pvals)
        siglags = (pcorr < 0.01).nonzero()[0]
        yheight = ax.get_ylim()[1]
        ax.scatter(lags[siglags], [yheight]*len(siglags), marker='*', color='blue')

    fig.savefig(f'results/figures/podcast-0shot-{experiment}-n_{nelecs}.png')
