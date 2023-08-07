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

    results = [  # two interactions (usually first glove and then gpt2)
        # "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
        "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_precentral-n_46.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-717_NYU_class_IFG-n_41.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-742_NYU_class_IFG-n_14.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-798_NYU_class_IFG-n_26.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv",
        "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_precentral-n_46.csv",
    ]
    titles = [
        # "Symbolic-PCA",  # symbolic with pca
        # "Symbolic",  # symbolic
        # "PCA",  # glove
        # "No-PCA",  # gpt2
        # "Ridge",  # gpt2
        # "Sym-1",
        # "Sym-3",
        # "Sym-7",
        # "Sym-3",
        # "Sym-Arb",
        # "Sym",
        # "Sym-Arb-C",
        # "Sym-C",
        # "Sym-0hot",
        # "Sym10",
        # "Sym20",
        # "Sym10",
        # "Static",
        # "C5",
        # "OldC5",
        # "C10",
        # "OldC10",
        # "C20",
        # "OldC20",
        # "GPT2-n",
        # "Sym-8",
        # "Sym-9",
        # "All",
        # "Arb",
        # "POS",
        # "Stop",
        # "Shape",
        # "Prefix",
        # "Suffix",
        "GloVe",
        "GPT-2",
        # "Participant 1",
        # "Participant 2",
        # "Participant 3",
    ]
    assert len(titles) == len(results)

    fig, axes = plt.subplots(1, len(results), figsize=(40, 15))
    # turn mitchell on for near-neighbor test analysis :)
    mitchell = True
    mitchell = False

    for ax, file, plt_title in zip(np.ravel(axes), results, titles):
        ax.set_ylim(-0.05, 0.16)
        ax.set_ylim(-0.05, 0.1)
        ax.set_xlim(-4000, 4000)
        # ax.set_xlabel("Time (ms)", fontsize=40, fontweight="bold")
        # ax.set_ylabel("Correlation", fontsize=40, fontweight="bold")
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
            fontsize=50,
        )
        # ax.set_yticks([-0.05, 0, 0.05, 0.1, 0.15])
        ax.set_yticks([-0.05, 0, 0.05, 0.1])
        # yticklabels = ["-0.05", "0", "0.05", "0.1", "0.15"]
        yticklabels = ["-0.05", "0", "0.05", "0.1"]
        ax.set_yticklabels(
            yticklabels,
            weight="bold",
            fontsize=50,
        )

        # ax.set_title(plt_title, fontsize=70, fontweight="bold")

        lags = np.arange(-4000, 4025, 25)
        lw = 5
        alpha = 0.2

        df = pd.read_csv(file)

        # Plot Blue line (regular encoding)
        df_actual = df[df.type == "actual"]
        ax.axhline(df_actual.threshold.unique()[0], color="blue", linewidth=3)
        df_actual.drop(
            columns={"Unnamed: 0", "type", "threshold"}, inplace=True
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
        df_nn.drop(columns={"Unnamed: 0", "type", "threshold"}, inplace=True)
        means = df_nn.mean()
        sems = df_nn.sem()
        ax.plot(lags, means, color=col, lw=lw)
        ax.fill_between(
            lags, means - sems, means + sems, alpha=alpha, color=col
        )

        # Plot Shuffle
        df_sh = df[df.type == "shuffle"]
        df_sh.drop(columns={"Unnamed: 0", "type", "threshold"}, inplace=True)
        means = df_sh.mean()
        sems = df_sh.sem()
        ax.plot(lags, means, color="black", lw=lw)
        ax.fill_between(
            lags, means - sems, means + sems, alpha=alpha, color="black"
        )

        # Plot Regular vs Near Neighbor Test Asterisks
        if "-nnt" in file and mitchell:
            nn_type = "is_significant_test"
        else:
            nn_type = "is_significant"
        df_sig = df[df.type == nn_type]
        df_sig.drop(columns={"Unnamed: 0", "type", "threshold"}, inplace=True)
        yheight = ax.get_ylim()[1] - 0.007
        sigs = df_sig.iloc[0, :].to_numpy().nonzero()[0]
        ax.scatter(
            lags[sigs], [yheight] * len(sigs), marker="*", color="blue", s=200
        )

        # Difference Test Asterisks
        if "glove-" in file or "symbolic-" in file:
            df_actual.reset_index(drop=True, inplace=True)
            df_nn.reset_index(drop=True, inplace=True)
            df_diff = df_actual - df_nn
        else:
            df_actual.reset_index(drop=True, inplace=True)
            df_nn.reset_index(drop=True, inplace=True)
            df_diff2 = df_actual - df_nn

            interactions = []
            for i in np.arange(df_diff.shape[1]):
                p = ttest_ind(df_diff2.iloc[:, i], df_diff.iloc[:, i])[1]
                interactions.append(p)
            interactions = fdr(interactions)
            interactions[interactions >= 0.01] = 0
            sigs = interactions.nonzero()[0]
            ax.scatter(
                lags[sigs],
                [yheight + 0.005] * len(sigs),
                marker="*",
                color="lightgreen",
                s=200,
            )

    fig.savefig("results/figures/plots/result.png")

    return


if __name__ == "__main__":
    main()
