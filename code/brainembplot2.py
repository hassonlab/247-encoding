
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import ttest_ind
from statsmodels.stats import multitest
from brainembplot import fdr


def main():

    # Used for interaction plot after running brainembplot.py
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    results = [ # two interactions (usually first glove and then gpt2)
        "results/figures/glove-concat-n2/0shot-dat-0-6-all3_NYU_class_IFG-n_81.csv",
        "results/figures/gpt2-fold-aligned/0shot-dat-d-2-all3_NYU_class_IFG-n_81.csv"
    ]

    for ax, file in zip(axes, results):
        ax.set_ylim(-0.05, 0.1)
        ax.set_xlim(-4000, 4000)
        ax.set_xlabel("Time (ms)", fontsize=15,fontweight="bold")
        ax.set_ylabel("Correlation", fontsize=15,fontweight="bold")
        ax.tick_params(labelsize=14)
        xticklabels = ["-4000","-3000","-2000","-1000","0","1000","2000","3000","4000"]
        ax.set_xticklabels(xticklabels, rotation=30, ha='right', rotation_mode='anchor',weight="bold")
        ax.set_yticks([-0.05, 0, 0.05, 0.1])
        yticklabels = ["-0.05", "0", "0.05", "0.1"]
        ax.set_yticklabels(yticklabels, weight="bold")

        if "glove" in file:
            title = "GloVe"
        else:
            title = "GPT2"
        ax.set_title(title,fontsize=40,fontweight="bold")

        lags = np.arange(-4000,4025,25)

        df = pd.read_csv(file)
    
        df_actual = df[df.type == "actual"]
        ax.axhline(df_actual.threshold.unique()[0], color="blue",linewidth=3)
        df_actual.drop(columns={"Unnamed: 0","type","threshold"},inplace=True)
        means = df_actual.mean()
        sems = df_actual.sem()
        ax.plot(lags, means, color="blue")
        ax.fill_between(lags, means - sems, means + sems, alpha=0.1, color="blue")

        df_nn= df[df.type == "near_neighbor"]
        df_nn.drop(columns={"Unnamed: 0","type","threshold"},inplace=True)
        means = df_nn.mean()
        sems = df_nn.sem()
        ax.plot(lags, means, color="red")
        ax.fill_between(lags, means - sems, means + sems, alpha=0.1, color="red")

        df_sh = df[df.type == "shuffle"]
        df_sh.drop(columns={"Unnamed: 0","type","threshold"},inplace=True)
        means = df_sh.mean()
        sems = df_sh.sem()
        ax.plot(lags, means, color="black")
        ax.fill_between(lags, means - sems, means + sems, alpha=0.1, color="black")

        df_sig = df[df.type == "is_significant"]
        df_sig.drop(columns={"Unnamed: 0","type","threshold"},inplace=True)
        yheight = ax.get_ylim()[1] - 0.005
        sigs = df_sig.iloc[0,:].to_numpy().nonzero()[0]
        ax.scatter(lags[sigs], [yheight] * len(sigs), marker = "*", color="blue")

        if "glove" in file:
            df_actual.reset_index(drop=True,inplace=True)
            df_nn.reset_index(drop=True,inplace=True)
            df_diff = df_actual - df_nn
        else:
            df_actual.reset_index(drop=True,inplace=True)
            df_nn.reset_index(drop=True,inplace=True)
            df_diff2 = df_actual - df_nn

            interactions = []
            for i in np.arange(df_diff.shape[1]):
                p = ttest_ind(df_diff2.iloc[:,i],df_diff.iloc[:,i])[1]
                interactions.append(p)
            interactions = fdr(interactions)
            interactions[interactions>=0.01] = 0
            sigs = interactions.nonzero()[0]
            ax.scatter(lags[sigs], [yheight + 0.004] * len(sigs), marker="*",color="lightgreen")

    fig.savefig("results/figures/interaction.png")

    return


if __name__ == "__main__":
    main()