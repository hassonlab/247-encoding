import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import ttest_ind
from statsmodels.stats import multitest


def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(
        pvals, method="fdr_bh", is_sorted=False
    )
    return pcor


def main():
    # Used for interaction plot after running brainembplot.py

    # results = [  # two interactions (usually first glove and then gpt2)
    #     # "results/figures/symbolic-1/0shot-dat-c-1-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-2/0shot-dat-c-2-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-4/0shot-dat-c-4-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-5/0shot-dat-c-5-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-6/0shot-dat-c-6-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-1-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-2-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-4-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-5-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-6-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3/0shot-dat-c-3-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-test/0shot-dat-est-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca1/0shot-dat-at1-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca2/0shot-dat-at2-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca4/0shot-dat-at4-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca5/0shot-dat-at5-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca10/0shot-dat-t10-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-concatpca20/0shot-dat-t20-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-3-ridge/0shot-dat-3-r-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-8/0shot-dat-c-8-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-11/0shot-dat--11-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-12/0shot-dat--12-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7/0shot-dat-c-7-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-test/0shot-dat-est-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca1/0shot-dat-at1-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca2/0shot-dat-at2-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca4/0shot-dat-at4-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca5/0shot-dat-at5-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca10/0shot-dat-t10-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-7-concatpca20/0shot-dat-t20-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-9/0shot-dat-c-9-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10/0shot-dat--10-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-concat3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br3/0shot-dat--br-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br5/0shot-dat-br5-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-arb-2/0shot-dat-arb-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-pos-2/0shot-dat-pos-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-stop-2/0shot-dat-top-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-shape-2/0shot-dat-ape-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-prefix-2/0shot-dat-fix-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/symbolic-10-br-suffix-2/0shot-dat-fix-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-test/0shot-dat-est-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca1/0shot-dat-at1-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca2/0shot-dat-at2-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca4/0shot-dat-at4-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca5/0shot-dat-at5-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-old/glove-concat5/0shot-dat-5-6-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca10/0shot-dat-t10-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/glove-concat10/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/glove-1-concatpca20/0shot-dat-t20-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-old/glove-concat20/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/gpt2-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/gpt2-test/0shot-dat-est-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures/gpt2-new/0shot-dat-est-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/gpt2-fold-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
    #     # "results/figures-2nd/gpt2-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
    # ]

    results = [
        # "results/figures-3rd/symbolic/0shot-dat-c-9-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/symbolic-9-3/0shot-dat-9-3-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/symbolic-9-4/0shot-dat-9-4-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/symbolic-9-5/0shot-dat-9-5-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/symbolic-9-9-4/0shot-dat-9-4-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-concat1/0shot-dat-at1-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-concat2/0shot-dat-at2-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-concat3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-nopca/0shot-dat-pca-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-nopca-concat1/0shot-dat-at1-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-nopca-concat2/0shot-dat-at2-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-nopca-concat3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/glove-9-4/0shot-dat-9-4-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd-old/glove-concat5/0shot-dat-5-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/glove-concat10/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd-old/glove-concat20/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-fold-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd-old/gpt2-9-4/0shot-dat-9-4-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv",
    ]
    titles = [
        # "Arb",  # arbitrary
        # "Arb-1",  # arbitrary-concats
        # "Arb-2",  # arbitrary-concats
        # "Arb-3",  # arbitrary-concats
        # "Sym",  # symbolic
        # "Sym",  # symbolic
        # "Sym-1",  # symbolic-concats
        # "Sym-2",  # symbolic-concats
        # "Sym-3",  # symbolic-concats
        # "ArbSym",  # arb sym
        # "ArbSym-1",  # arb sym
        # "ArbSym-2",  # arb sym
        # "ArbSym-3",  # arb sym
        # "GloVe",  # gpt2
        # "GloVe20",  # gpt2
        # "GPT2",  # gpt2
        # "Arb",
        # "POS",
        # "Stop",
        # "Shape",
        # "Prefix",
        # "Suffix",
    ]
    assert len(titles) == len(results)

    fig, axes = plt.subplots(1, len(results), figsize=(30, 12))
    # turn mitchell on for near-neighbor test analysis :)
    mitchell = True
    mitchell = False

    side_label = True
    for ax, file, plt_title in zip(np.ravel(axes), results, titles):
        ax.set_ylim(-0.05, 0.1)
        ax.set_xlim(-4000, 4000)
        ax.set_xlabel("Time (ms)", fontsize=30, fontweight="bold")
        if side_label:
            ax.set_ylabel("Correlation", fontsize=30, fontweight="bold")
            side_label = True
        ax.tick_params(labelsize=14)
        xticklabels = [
            "-4000",
            "-3000",
            "-2000",
            "-1000",
            "0",
            "1000",
            "2000",
            "3000",
            "4000",
        ]
        ax.set_xticklabels(
            xticklabels,
            rotation=30,
            ha="right",
            rotation_mode="anchor",
            weight="bold",
            fontsize=23,
        )
        ax.set_yticks([-0.05, 0, 0.05, 0.1])
        yticklabels = ["-0.05", "0", "0.05", "0.1"]
        ax.set_yticklabels(
            yticklabels,
            weight="bold",
            fontsize=23,
        )

        # ax.set_title(plt_title, fontsize=40, fontweight="bold")

        lags = np.arange(-4000, 4025, 25)

        df = pd.read_csv(file)
        lw = 4
        alpha = 0.1
        alpha = 0.15

        # Plot Blue line (regular encoding)
        df_actual = df[df.type == "actual"]
        # ax.axhline(df_actual.threshold.unique()[0], color="blue", linewidth=3)
        df_actual.drop(
            columns={"Unnamed: 0", "type", "threshold"},
            errors="ignore",
            inplace=True,
        )
        means = df_actual.mean()
        sems = df_actual.sem()
        ax.plot(lags, means, color="blue", lw=lw)
        ax.fill_between(
            lags, means - sems, means + sems, alpha=alpha, color="blue"
        )

        # Plot Near Neighbor
        if "-nnt" in file and mitchell:
            nn_type = "near_neighbor_test"
            col = "fuchsia"
        else:
            nn_type = "near_neighbor"
            col = "red"
        df_nn = df[df.type == nn_type]
        df_nn.drop(
            columns={"Unnamed: 0", "type", "threshold"},
            errors="ignore",
            inplace=True,
        )
        means = df_nn.mean()
        sems = df_nn.sem()
        ax.plot(lags, means, color=col, lw=lw)
        ax.fill_between(
            lags, means - sems, means + sems, alpha=alpha, color=col
        )

        # Plot Shuffle
        df_sh = df[df.type == "shuffle"]
        df_sh.drop(
            columns={"Unnamed: 0", "type", "threshold"},
            errors="ignore",
            inplace=True,
        )
        means = df_sh.mean()
        sems = df_sh.sem()
        ax.plot(lags, means, color="black", lw=lw)
        ax.fill_between(
            lags, means - sems, means + sems, alpha=alpha, color="black"
        )

        # # Plot Regular vs Near Neighbor Test Asterisks
        # if "-nnt" in file and mitchell:
        #     nn_type = "is_significant_test"
        # else:
        #     nn_type = "is_significant"
        # df_sig = df[df.type == nn_type]
        # df_sig.drop(
        #     columns={"Unnamed: 0", "type", "threshold"},
        #     errors="ignore",
        #     inplace=True,
        # )
        # yheight = ax.get_ylim()[1] - 0.005
        # sigs = df_sig.iloc[0, :].to_numpy().nonzero()[0]
        # ax.scatter(lags[sigs], [yheight] * len(sigs), marker="*", color="blue")

        # Difference Test Asterisks
        # if "glove-" in file or "symbolic-" in file:
        #     df_actual.reset_index(drop=True, inplace=True)
        #     df_nn.reset_index(drop=True, inplace=True)
        #     df_diff = df_actual - df_nn
        # else:
        #     df_actual.reset_index(drop=True, inplace=True)
        #     df_nn.reset_index(drop=True, inplace=True)
        #     df_diff2 = df_actual - df_nn

        #     interactions = []
        #     for i in np.arange(df_diff.shape[1]):
        #         p = ttest_ind(df_diff2.iloc[:, i], df_diff.iloc[:, i])[1]
        #         interactions.append(p)
        #     interactions = fdr(interactions)
        #     interactions[interactions >= 0.01] = 0
        #     sigs = interactions.nonzero()[0]
        #     ax.scatter(
        #         lags[sigs],
        #         [yheight + 0.004] * len(sigs),
        #         marker="*",
        #         color="lightgreen",
        #     )

    fig.savefig("results/figures/plots/result.png")

    return


if __name__ == "__main__":
    main()
