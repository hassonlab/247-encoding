import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tfsenc_read_datum import (
    load_datum,
    drop_nan_embeddings,
    remove_punctuation,
)


def run_pca(df):
    pca = PCA(n_components=50, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


def main():
    # sids = [625, 676, 7170, 798]
    # embs = {"gpt2": [70, 12], "gpt2-medium": [70, 24]}
    sids = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512]
    embs = {"whisper-medium.en": [1, 24]}
    for emb in embs.keys():
        cnxt = embs[emb][0]
        layer = embs[emb][1]

        for sid in sids:
            dir = f"data/tfs/{sid}/pickles/embeddings/{emb}/full"
            base_df_path = os.path.join(dir, "base_df.pkl")
            emb_df_path = os.path.join(dir, f"cnxt_{cnxt:04}/layer_{layer:02}.pkl")

            base_df = load_datum(base_df_path)
            emb_df = load_datum(emb_df_path)

            df = pd.merge(base_df, emb_df, left_index=True, right_index=True)
            print(sid, len(base_df), len(emb_df), len(df))

            # df = drop_nan_embeddings(df)
            # df = remove_punctuation(df)
            # df = run_pca(df)
            # df.to_csv(f"tfs-{sid}-{emb}-cnxt{cnxt}-layer{layer}.csv")
            # df.to_csv(f"sub-{str(sid)[1:]}_{emb}-layer{layer}.csv")
            df.to_pickle(f"results/for-zaid/sub-{str(sid)[1:]}_{emb}-layer{layer}.pkl")

    return


if __name__ == "__main__":
    main()
