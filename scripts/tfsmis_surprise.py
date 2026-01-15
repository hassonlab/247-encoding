import numpy as np
import pandas as pd
from tfsenc_read_datum import load_datum
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    sids = [625, 676, 7170, 798]

    all_df = pd.DataFrame()
    for idx, sid in enumerate(sids):
        sent_df = pd.read_csv(f"results/tfs/npvp/{sid}_sent_df_constituency_labels.csv")
        datum = pd.read_pickle(f"{sid}_datum_preds.pkl")
        datum["production"] = datum.speaker.str.contains("Speaker1")
        datum["production"] = datum.production.map({True: f"S{idx+1}", False: "Other"})

        df = datum.merge(sent_df, on=["word", "adjusted_onset"], how="left")
        df = df.loc[~df.phrase_start.isna()]
        all_df = pd.concat([all_df, df], ignore_index=True)

    all_df["phrase_group"] = "middle"
    all_df.loc[all_df.phrase_start, "phrase_group"] = "start"
    all_df.loc[all_df.phrase_end, "phrase_group"] = "end"
    all_df.loc[all_df.phrase_start & all_df.phrase_end, "phrase_group"] = "single"
    all_df["sent_group"] = "middle"
    all_df.loc[all_df.label_idx.eq(1), "sent_group"] = "start"
    all_df.loc[all_df.label_idx_r.eq(1), "sent_group"] = "end"
    all_df.loc[all_df.label_idx.eq(1) & all_df.label_idx_r.eq(1), "sent_group"] = (
        "single"
    )

    breakpoint()

    sns.set_style("whitegrid")
    snsplt = sns.lineplot(
        data=all_df,
        x="production",
        y="true_pred_prob",
        # hue="sent_group",
        hue="phrase_group",
        marker="o",
        markersize=10,
        # alpha = 0.5,
        linestyle="",
        err_style="bars",
        errorbar="se",
        hue_order=["start", "middle", "end", "single"],
    )
    plt.savefig(f"tfsmis_surprise_phrase.png")
    breakpoint()
    df_mid = df.loc[df.phrase_start.ne(True) & df.phrase_end.ne(True)]

    return


if __name__ == "__main__":
    main()
