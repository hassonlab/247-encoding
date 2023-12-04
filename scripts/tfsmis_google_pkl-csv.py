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


def read_datum2(base_df_path, emb_df_path):
    """Load, process, and filter datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        DataFrame: processed and filtered datum
    """
    base_df = load_datum(base_df_path)
    emb_df = load_datum(emb_df_path)
    df = pd.concat([base_df, emb_df], axis=1)

    print(f"After loading: Datum loads with {len(df)} words")

    df = df[df.adjusted_onset.notna()]
    df = drop_nan_embeddings(df)
    print(f"After processing: Datum now has {len(df)} words")

    print("Taking only full 30s utterance chunks")
    df = df[df.window_num == 1500]

    print("Taking full windows only")
    df = df[df.full_window == 1]  # only take full windows
    print("Shifting onset to mid of window")
    df["adjusted_onset"] = df.adjusted_onset + 0.0325 * 512
    print(f"After filtering: Datum now has {len(df)} words")

    print(f"Selecting non-overlapping windows")
    df = df[df.window_idx % 3 == 2]

    df["adjusted_onset"] = df.adjusted_onset.astype("float64")
    df["adjusted_offset"] = df.adjusted_offset.astype("float64")
    df = run_pca(df)
    print(f"Final datum length: {len(df)}")

    return df


def main():
    sids = [798, 676]
    sids = [625, 7170, 798, 676]
    embs = {"whisper-tiny.en-acoustic": [1, 4]}
    for emb in embs.keys():
        cnxt = embs[emb][0]
        layer = embs[emb][1]

        for sid in sids:
            dir = f"data/tfs/{sid}/pickles/embeddings/{emb}/full"
            base_df_path = os.path.join(dir, "base_df.pkl")
            emb_df_path = os.path.join(dir, f"cnxt_{cnxt:04}/layer_{layer:02}.pkl")

            df = read_datum2(base_df_path, emb_df_path)
            df.to_csv(f"tfs-{sid}-{emb}-cnxt{cnxt}-layer{layer}.csv")

    return


if __name__ == "__main__":
    main()
