import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


pca_to = 50

def run_pca(pca_to, df):
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


embs = [    '/scratch/gpfs/ln1144/247-pickling/results/tfs/625/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/cnxt_0001/layer_03.pkl',
            '/scratch/gpfs/ln1144/247-pickling/results/tfs/676/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/cnxt_0001/layer_03.pkl',
            '/scratch/gpfs/ln1144/247-pickling/results/tfs/798/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/cnxt_0001/layer_03.pkl',
            '/scratch/gpfs/ln1144/247-pickling/results/tfs/7170/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/cnxt_0001/layer_03.pkl'
]

base = [   '/scratch/gpfs/ln1144/247-pickling/results/tfs/625/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/base_df.pkl',
           '/scratch/gpfs/ln1144/247-pickling/results/tfs/676/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/base_df.pkl',
           '/scratch/gpfs/ln1144/247-pickling/results/tfs/798/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/base_df.pkl',
           '/scratch/gpfs/ln1144/247-pickling/results/tfs/7170/pickles/embeddings/whisper-tiny.en-decoder-corrected/full/base_df.pkl'
]

out_paths = ['/scratch/gpfs/ln1144/247-encoding/data/tfs/embeddings_google_pca50/625_whisper_de_best_correct.csv',
             '/scratch/gpfs/ln1144/247-encoding/data/tfs/embeddings_google_pca50/676_whisper_de_best_correct.csv',
             '/scratch/gpfs/ln1144/247-encoding/data/tfs/embeddings_google_pca50/798_whisper_de_best_correct.csv',
             '/scratch/gpfs/ln1144/247-encoding/data/tfs/embeddings_google_pca50/7170_whisper_de_best_correct.csv',
             
]


for i in range(0,len(embs)):

    # breakpoint()

    emb_df = pd.read_pickle(embs[i])
    emb_df = pd.DataFrame(emb_df)

    # get df with embeddings pca'd
    emb_df = run_pca(pca_to,emb_df)

    # breakpoint()
    
    # get base_dfs, append embeddings
    base_df = pd.read_pickle(base[i])
    base_df = pd.DataFrame(base_df)

    assert(len(emb_df.index) == len(base_df.index))
    out_df = base_df.join(emb_df)

    # and save to csv
    out_df.to_csv(out_paths[i],header=True,index=True)

