# e22 - rerran 10 folds, pc
# e23 - same but 4s

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool

from scipy.stats import zscore
from statsmodels.stats import multitest

# from statsmodels.stats import stattools

n_workers = 4
nperms = 0
pdir = "/scratch/gpfs/kw1166/0shot-encoding/results/podcast/"

# dirs = [
#     "0shot-zz-podcast-full-777-gpt2-xl-e23/777/",
#     "0shot-zz-podcast-full-777-gpt2-xl-e23-sh/777/",
# ]

# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717l/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717lfdp/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717fdp/777/']
# dirs = ['0shot-zz-podcast-full-777-gpt2-xl-717detrend/777/']
# dirs = ["0shot-zz-podcast-full-777-gpt2-xl-717ldp1/777/"]

# dirs = [
#     "0shot-kw-podcast-full-777-gpt2-xl-shift-emb/777/",
#     "0shot-kw-podcast-full-777-gpt2-xl-shift-emb-sh/777/",
# ]

# dirs = [
#     "0shot-kw-podcast-full-777-gpt2-xl-48-test/777/",
#     "0shot-kw-podcast-full-777-gpt2-xl-fold-aligned-sh/777/",
# ]

dirs = [
    "0shot-kw-podcast-full-777-symbolic-9-9-4/777/",
    "0shot-kw-podcast-full-777-symbolic-9-sh/777/",
]

# dirs = [
#     "0shot-kw-podcast-full-777-glove50-1-test/777/",
#     "0shot-kw-podcast-full-777-glove50-aligned-sh/777/",
# ]

# Create experiments from master list
elecs = pd.read_csv("data/elec_masterlist.csv")
cats = ["princeton_class", "NYU_class"]

cats = ["NYU_class"]
subjects = [
    717,
    798,
    742,
    [717, 798, 742],
    # [662, 717, 723, 741, 742, 743, 763, 798],
]
# subjects = [[717, 798, 742]]
rois = ["IFG"]
rois = ["IFG", "precentral", "postcentral", "STG"]

experiments = {}
for category in cats:
    for subject in subjects:
        for roi in rois:
            if isinstance(subject, int):
                crit = elecs.subject == subject
                name = "_".join([str(subject), category, roi])
            elif isinstance(subject, list):
                m = len(subject)
                crit = elecs.subject.isin(subject)
                name = "_".join([f"all{m}", category, roi])
            crit &= elecs[category] == roi
            subdf = elecs[crit]
            es = [str(x) + "_" + y for x, y in zip(subdf.subject, subdf.name)]
            if len(es):
                experiments[name] = es

custom = dirs[0].split("/")[0][-3:]
print(custom)
print(len(experiments), "experiments")
if False:
    # save experiments to different csvs
    for experiment in experiments:
        print(experiment, len(experiments[experiment]))
        region_elecs = pd.DataFrame(experiments[experiment])
        region_elecs[["subject", "electrode"]] = region_elecs[0].str.split(
            "_", expand=True
        )
        region_elecs = region_elecs.loc[:, ("subject", "electrode")]
        region_elecs.to_csv(f"{experiment}.csv", index=False)
# if elec == '717_LGB79':


def correlate(A, B, axis=0):
    """Calculate pearson correlation between two matricies.

    axis = 0 correlates columns in A to columns in B
    axis = 1 correlates rows in A to rows in B
    """
    assert A.ndim == B.ndim, "Matrices must have same number of dimensions"
    assert A.shape == B.shape, "Matrices must have same shape"

    A_mean = A.mean(axis=axis, keepdims=True)
    B_mean = B.mean(axis=axis, keepdims=True)
    A_stddev = np.sum((A - A_mean) ** 2, axis=axis)
    B_stddev = np.sum((B - B_mean) ** 2, axis=axis)

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
        dist[i] = np.mean(s * (x - y))

    p_value = (truescore > dist).mean()
    return p_value


def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(
        pvals, method="fdr_bh", is_sorted=False
    )
    return pcor


def run_exp(experiment, elecs):
    print(experiment, len(elecs))

    dfs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(0, ls="-", c="black", alpha=0.1)
    ax.axhline(0, ls="-", c="black", alpha=0.1)
    ax.set(xlabel="Lag (s)", ylabel="Correlation (r+se)")
    # ax.set_ylim([-0.05, 0.25])

    for i, resultdir in enumerate(dirs_new):
        lags = []
        signal = []
        pred_signal = []
        nn_signal = []
        nnt_signal = []

        # Load results of this run
        for elec in elecs:
            if pd.isna(elec):
                continue
            # filename = pdir + resultdir + elec[2:5] + '_' + elec[30:] + '.pkl'
            filename = pdir + resultdir + elec + ".pkl"
            if not os.path.isfile(filename):
                print(filename, "is not found")
                continue
            with open(filename, "rb") as f:
                # print(f"Loading{filename}")
                data = pickle.load(f)
                lags = data["lags"]
                signal.append(data["Y_signal"])
                pred_signal.append(data["Yhat_signal"])
                nn_signal.append(data["Yhat_nn_signal"])
                nnt_signal.append(data["Yhat_nnt_signal"])

        if len(signal) == 0:
            print("None of the electrodes were found")
            break

        print(f"Found {len(signal)} electrodes for experiment {experiment}")

        signal = np.stack(signal, axis=-1)  # n_words x n_lags x n_elecs
        pred_signal = np.stack(pred_signal, axis=-1)
        nn_signal = np.stack(nn_signal, axis=-1)
        nnt_signal = np.stack(nnt_signal, axis=-1)

        corrs, corrs_nn, corrs_nnt = [], [], []
        sems, sems_nn, sems_nnt = [], [], []
        rawcorr, rawcorr_nn, rawcorr_nnt = [], [], []
        acdw = []

        for lag in range(signal.shape[1]):
            A = signal[:, lag, :]  # n_words x n_elecs
            B = pred_signal[:, lag, :]  # n_words x n_elecs
            C = nn_signal[:, lag, :]  # n_words x n_elecs
            D = nnt_signal[:, lag, :]  # n_words x n_elecs
            # acdw.append(stattools.durbin_watson(A - B, axis=0).mean())
            A = zscore(A, axis=0)
            B = zscore(B, axis=0)
            C = zscore(C, axis=0)
            D = zscore(D, axis=0)

            # correlation of original & predicted
            rs = correlate(A, B, axis=1)  # 1 is rows, 0 is columns
            corrs.append(rs.mean())
            sems.append(rs.std() / np.sqrt(len(rs)))
            rawcorr.append(rs)

            # correlation of original & nn-train predicted
            rs = correlate(A, C, axis=1)
            corrs_nn.append(rs.mean())
            sems_nn.append(rs.std() / np.sqrt(len(rs)))
            rawcorr_nn.append(rs)

            # correlation of original & nn-test predicted
            rs = correlate(A, D, axis=1)
            corrs_nnt.append(rs.mean())
            sems_nnt.append(rs.std() / np.sqrt(len(rs)))
            rawcorr_nnt.append(rs)

        # print('avg DW:', np.mean(acdw))
        nelecs = signal.shape[-1]

        # plot for original correlation
        # lags = list(map(str, lags))
        lags = np.asarray(lags) / 1000
        xaxis = lags
        mean = np.asarray(corrs)
        err = np.asarray(sems)
        col = (
            "blue" if i == 0 else "gray"
        )  # blue for original, grey for shuffled
        ax.plot(xaxis, mean, color=col)
        ax.fill_between(xaxis, mean - err, mean + err, alpha=0.1, color=col)

        corrs = np.vstack(rawcorr)
        corrs2 = np.vstack(rawcorr_nn)
        corrs3 = np.vstack(rawcorr_nnt)

        df = pd.DataFrame(corrs.T, columns=lags)
        df.insert(0, "type", "actual" if i == 0 else "shuffle")
        dfs.append(df)

        if i == 0:
            # plot for red line (nn-train cor)
            mean_nn = np.asarray(corrs_nn)
            err_nn = np.asarray(sems_nn)
            ax.plot(xaxis, mean_nn, color="red")
            ax.fill_between(
                xaxis,
                mean_nn - err_nn,
                mean_nn + err_nn,
                alpha=0.1,
                color="red",
            )

            df = pd.DataFrame(corrs2.T, columns=lags)
            df.insert(0, "type", "near_neighbor")
            dfs.append(df)

            # plot for fuchsia line (nn-test cor)
            mean_nnt = np.asarray(corrs_nnt)
            err_nnt = np.asarray(sems_nnt)
            ax.plot(xaxis, mean_nnt, color="orange")
            ax.fill_between(
                xaxis,
                mean_nnt - err_nnt,
                mean_nnt + err_nnt,
                alpha=0.1,
                color="orange",
            )

            df = pd.DataFrame(corrs3.T, columns=lags)
            df.insert(0, "type", "near_neighbor_test")
            dfs.append(df)

        ax.set_title(f"{experiment} | N={nelecs}")

        if i == 0 and nperms > 0:
            # Calculate max of correlation significance
            pvals = [one_samp_perm(corrs[i], nperms) for i in range(len(lags))]
            pcorr = fdr(pvals)
            m = corrs.mean(axis=1)
            g = pcorr <= 0.01
            minP = 0
            if g.any():
                sigSig = m[g]
                if (pcorr > 0.01).any():
                    maxP = m[pcorr > 0.01].max()
                    gg = sigSig > maxP
                    if gg.any():
                        minP = sigSig[gg].min()
                        ax.axhline(minP)

            # sig test for nn-train
            pvals = [
                paired_permutation(corrs2[i], corrs[i], nperms)
                for i in range(len(lags))
            ]
            pcorr = fdr(pvals)
            issig = (m > minP) & (pcorr < 0.01)
            siglags = issig.nonzero()[0]
            yheight = ax.get_ylim()[1] - 0.005
            ax.scatter(
                lags[siglags],
                [yheight] * len(siglags),
                marker="*",
                color="blue",
            )

            dfs[0].insert(1, "threshold", minP)
            df = pd.DataFrame(issig).T.set_axis(lags, axis=1)
            df.insert(0, "type", "is_significant")
            dfs.append(df)

            # sig test for nn-test
            pvals = [
                paired_permutation(corrs3[i], corrs[i], nperms)
                for i in range(len(lags))
            ]
            pcorr = fdr(pvals)
            issig = (m > minP) & (pcorr < 0.01)
            # siglags = issig.nonzero()[0]
            # yheight = ax.get_ylim()[1] - .005
            # ax.scatter(
            #     lags[siglags],
            #     [yheight]*len(siglags),
            #     marker='*',
            #     color='orange',
            # )

            df = pd.DataFrame(issig).T.set_axis(lags, axis=1)
            df.insert(0, "type", "is_significant_test")
            dfs.append(df)

        # # breakpoint()
        # from matplotlib.backends.backend_pdf import PdfPages
        # pdf = PdfPages('tmp.pdf')
        # fig, ax = plt.subplots()
        # ax.axvline(0, ls='-', c='black', alpha=0.1)
        # ax.axhline(0, ls='-', c='black', alpha=0.1)
        # ax.set(xlabel='Lag (s)', ylabel='Correlation (r+se)')
        # # ax.set_ylim([-0.05, 0.25])
        # ax.plot(lags, corrs.mean(axis=1), color='blue')
        # ax.plot(lags, corrs2.mean(axis=1), color='red')
        # ax.set_title('all')
        # pdf.savefig(fig)
        # plt.close()
        # for j in range(nelecs):
        #     fig, ax = plt.subplots()
        #     ax.axvline(0, ls='-', c='black', alpha=0.1)
        #     ax.axhline(0, ls='-', c='black', alpha=0.1)
        #     ax.set(xlabel='Lag (s)', ylabel='Correlation (r+se)')
        #     # ax.set_ylim([-0.05, 0.25])
        #     ax.plot(lags, corrs[:,j], color='blue')
        #     ax.plot(lags, corrs2[:,j], color='red')
        #     ax.set_title(elecs[j])
        #     pdf.savefig(fig)
        #     plt.close()
        # pdf.close()

    df = pd.concat(dfs)
    df.to_csv(f"results/figures/0shot-dat-{custom}-{experiment}-n_{nelecs}.csv")

    fig.savefig(
        f"results/figures/0shot-fig-{custom}-{experiment}-n_{nelecs}.png"
    )
    plt.close()


if __name__ == "__main__":
    dirs_new = dirs
    with Pool(min(n_workers, len(experiments))) as p:
        p.starmap(run_exp, experiments.items())

    # for idx in np.arange(1,62): # for looping
    #     custom = f"{idx:02}"
    #     dirs_new = dirs.copy()
    #     dirs_new[0] = dirs_new[0] % idx
    #     with Pool(min(n_workers, len(experiments))) as p:
    #         p.starmap(run_exp, experiments.items())
