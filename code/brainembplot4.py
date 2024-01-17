import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import ttest_ind
from statsmodels.stats import multitest


def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(pvals, method="fdr_bh", is_sorted=False)
    return pcor


def read_actual(filename):
    df = pd.read_csv(filename)
    df = df[df.type == "actual"]
    df.drop(
        columns={"Unnamed: 0", "type", "threshold"},
        errors="ignore",
        inplace=True,
    )

    return df


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
        # "results/figures-3rd/symbolic-oneout/0shot-dat-%s-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic/0shot-dat-c-9-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-3rd/symbolic-concat3-oneout/0shot-dat-%s-all3_NYU_class_IFG-n_81.csv",
        "results/figures-3rd/symbolic-concat3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd-old/glove-concat5/0shot-dat-5-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/glove-concat10/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd-old/glove-concat20/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-fold-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-aligned-n-nnt/0shot-dat-emb-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv",
        # "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv",
    ]

    plt_title = "Symbolic"
    plt_title = "Symbolic-Concat3"

    fig, axes = plt.subplots(1, 1, figsize=(12, 12))

    mitchell = False

    side_label = True
    ax = axes

    file = results[0]
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

    # ax.set_title(plt_title, fontsize=50, fontweight="bold")

    lags = np.arange(-4000, 4025, 25)

    lw = 4
    alpha = 0.1
    alpha = 0.15

    if "%s" in file:
        df_final = pd.DataFrame()
        df_final_mean = pd.DataFrame()
        df_final_sems = pd.DataFrame()
        for idx in np.arange(1, 62):
            df = pd.read_csv(file % f"{idx:02}")
            df_actual = df[df.type == "actual"]
            df_actual.drop(
                columns={"Unnamed: 0", "type", "threshold"},
                errors="ignore",
                inplace=True,
            )

            means = df_actual.mean()
            sems = df_actual.sem()

            df_final = pd.concat((df_final, df_actual))
            df_final_mean = pd.concat((df_final_mean, means.to_frame().T))
            df_final_sems = pd.concat((df_final_sems, sems.to_frame().T))
        means = df_final.mean()
        sems = df_final.sem()
        ax.plot(lags, means, color="blue", lw=lw)
        ax.fill_between(lags, means - sems, means + sems, alpha=alpha, color="blue")
        # means = df_final_mean.mean()
        # sems = df_final_sems.mean()
        # ax.plot(lags, means, color="orange", lw=lw)
        # ax.fill_between(
        #     lags, means - sems, means + sems, alpha=alpha, color="orange"
        # )

    else:
        # Plot Blue line (regular encoding)
        df = pd.read_csv(file)
        df_actual = df[df.type == "actual"]
        threshold = df_actual.threshold.unique()[0]
        ax.axhline(threshold, color="blue", linewidth=3)
        df_actual.drop(
            columns={"Unnamed: 0", "type", "threshold"},
            errors="ignore",
            inplace=True,
        )

        means = df_actual.mean()
        sems = df_actual.sem()
        ax.plot(
            lags,
            means,
            color="blue",
            lw=lw,
            label="0Shot",
        )
        ax.fill_between(lags, means - sems, means + sems, alpha=alpha, color="blue")

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
        ax.plot(
            lags,
            means,
            color=col,
            lw=lw,
            label="NN",
        )
        ax.fill_between(lags, means - sems, means + sems, alpha=alpha, color=col)
        breakpoint()
        # Plot Shuffle
        df_sh = df[df.type == "shuffle"]
        df_sh.drop(columns={"Unnamed: 0", "type", "threshold"}, inplace=True)
        means = df_sh.mean()
        sems = df_sh.sem()
        ax.plot(
            lags,
            means,
            color="black",
            lw=lw,
            label="Shuffled",
        )
        ax.fill_between(lags, means - sems, means + sems, alpha=alpha, color="black")

        # Plot Regular vs Near Neighbor Test Asterisks
        if "-nnt" in file and mitchell:
            nn_type = "is_significant_test"
        else:
            nn_type = "is_significant"
        df_sig = df[df.type == nn_type]
        df_sig.drop(columns={"Unnamed: 0", "type", "threshold"}, inplace=True)
        yheight = ax.get_ylim()[1] - 0.005
        sigs = df_sig.iloc[0, :].to_numpy().nonzero()[0]
        ax.scatter(lags[sigs], [yheight] * len(sigs), marker="*", color="blue")

        sig_test = True
        if sig_test:
            # sig test for gpt2
            second_file = "results/figures-2nd/gpt2-fold-aligned-nnt/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv"
            second_file = "results/figures-3rd/symbolic-concat3/0shot-dat-at3-all3_NYU_class_IFG-n_81.csv"
            second_file = (
                "results/figures-3rd/symbolic/0shot-dat-c-9-all3_NYU_class_IFG-n_81.csv"
            )
            df_second = read_actual(second_file)
            interactions = []
            for i in np.arange(df_actual.shape[1]):
                p = ttest_ind(df_second.iloc[:, i], df_actual.iloc[:, i])[1]
                interactions.append(p)
            interactions = fdr(interactions)
            interactions[interactions >= 0.01] = 0
            # interactions[df_actual.mean(axis=0) < threshold] = 0
            sigs = interactions.nonzero()[0]
            ax.scatter(
                lags[sigs],
                [yheight + 0.003] * len(sigs),
                marker="*",
                color="orange",
            )

            # sig test for glove
            # glove_file = "results/figures-2nd/glove-aligned-nnt/0shot-dat-ned-all3_NYU_class_IFG-n_81.csv"
            # glove_file = "results/figures-2nd/glove-concat10/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv"
            # glove_file = "results/figures-2nd-old/glove-concat5/0shot-dat-5-6-all3_NYU_class_IFG-n_81.csv"
            # df_glove = read_actual(glove_file)
            # interactions = []
            # for i in np.arange(df_actual.shape[1]):
            #     p = ttest_ind(df_glove.iloc[:, i], df_actual.iloc[:, i])[1]
            #     interactions.append(p)
            # interactions = fdr(interactions)
            # interactions[interactions >= 0.0001] = 0
            # sigs = interactions.nonzero()[0]
            # ax.scatter(
            #     lags[sigs],
            #     [yheight] * len(sigs),
            #     marker="*",
            #     color="red",
            # )

    fig.savefig("results/figures/plots/result.png")

    return


if __name__ == "__main__":
    main()
