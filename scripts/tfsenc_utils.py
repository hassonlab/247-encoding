import csv
import os
from functools import partial
from imp import C_EXTENSION
from multiprocessing import Pool

import mat73
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline


def encColCorr(CA, CB):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2

    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()

    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    return r, p, t


def cv_lm_003_prod_comp(args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag):
    if lag == -1:
        print("running regression")
    else:
        print("running regression with best_lag")

    if args.pca_to == 0:
        print(f"No PCA, emb_dim = {Xtes.shape[1]}")
    else:
        print(f"PCA from {Xtes.shape[1]} to {args.pca_to}")

    nSamps = Xtes.shape[0]
    nChans = Ytra.shape[1] if Ytra.shape[1:] else 1

    YHAT = np.zeros((nSamps, nChans))
    Ynew = np.zeros((nSamps, nChans))

    for i in range(0, args.fold_num):
        Xtraf, Xtesf = Xtra[fold_tra != i], Xtes[fold_tes == i]
        Ytraf, Ytesf = Ytra[fold_tra != i], Ytes[fold_tes == i]

        # Xtesf -= np.mean(Xtraf, axis=0)
        # Xtraf -= np.mean(Xtraf, axis=0)
        Ytesf -= np.mean(Ytraf, axis=0)
        Ytraf -= np.mean(Ytraf, axis=0)

        # Fit model
        if args.pca_to == 0 or "nopca" in args.datum_mod:
            model = make_pipeline(StandardScaler(), LinearRegression())
        else:
            model = make_pipeline(
                StandardScaler(),
                PCA(args.pca_to, whiten=True),
                LinearRegression(),
            )
        model.fit(Xtraf, Ytraf)

        if lag != -1:
            B = model.named_steps["linearregression"].coef_
            assert lag < B.shape[0], f"Lag index out of range"
            B = np.repeat(
                B[lag, :][np.newaxis, :], B.shape[0], 0
            )  # best-lag model
            model.named_steps["linearregression"].coef_ = B

        # Predict
        foldYhat = model.predict(Xtesf)

        Ynew[fold_tes == i, :] = Ytesf.reshape(-1, nChans)
        YHAT[fold_tes == i, :] = foldYhat.reshape(-1, nChans)

    return (YHAT, Ynew)


@jit(nopython=True)
def build_Y(
    onsets, convo_onsets, convo_offsets, brain_signal, lags, window_size
):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    half_window = round((window_size / 1000) * 512 / 2)

    # Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))
    Y1 = np.zeros((len(onsets), len(lags)))

    for lag in prange(len(lags)):

        lag_amount = int(lags[lag] / 1000 * 512)

        index_onsets = np.minimum(
            convo_offsets - half_window - 1,
            np.maximum(
                convo_onsets + half_window + 1,
                np.round_(onsets, 0, onsets) + lag_amount,
            ),
        )

        # index_onsets = np.round_(onsets, 0, onsets) + lag_amount

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # vec = brain_signal[np.array(
        #     [np.arange(*item) for item in zip(starts, stops)])]

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag] = np.mean(brain_signal[start:stop].reshape(-1))

    return Y1


def build_XY(args, datum, brain_signal):
    """[summary]

    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    X = np.stack(datum.embeddings).astype("float64")

    word_onsets = datum.adjusted_onset.values
    convo_onsets = datum.convo_onset.values
    convo_offsets = datum.convo_offset.values

    lags = np.array(args.lags)
    brain_signal = brain_signal.reshape(-1, 1)

    Y = build_Y(
        word_onsets,
        convo_onsets,
        convo_offsets,
        brain_signal,
        lags,
        args.window_size,
    )

    return X, Y


def encoding_mp_prod_comp(
    args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag
):
    if args.shuffle:
        np.random.shuffle(Ytra)
        np.random.shuffle(Ytes)

    # train lm and predict
    PY_hat, Y_new = cv_lm_003_prod_comp(
        args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, lag
    )

    rps = []
    # correlation for whole datum
    rp, _, _ = encColCorr(Y_new, PY_hat)
    rps.append(rp)

    # correlation per folds
    for i in np.unique(fold_tes).astype(int):
        rp_fold, _, _ = encColCorr(Y_new[fold_tes == i], PY_hat[fold_tes == i])
        rps.append(rp_fold)

    return rps


def run_regression(args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes):
    perm_prod = []
    for i in range(args.npermutations):
        result = encoding_mp_prod_comp(
            args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, -1
        )
        if args.model_mod and "best-lag" in args.model_mod:
            best_lag = np.argmax(np.array(result))
            print("switch to best-lag: " + str(best_lag))
            perm_prod.append(
                encoding_mp_prod_comp(
                    args, Xtra, Ytra, fold_tra, Xtes, Ytes, fold_tes, best_lag
                )
            )
        else:
            perm_prod.append(result)

    return perm_prod


def get_groupkfolds(datum, X, Y, fold_num=10):
    fold_cat = np.zeros(datum.shape[0])
    grpkfold = GroupKFold(n_splits=fold_num)
    folds = [
        t[1] for t in grpkfold.split(X, Y, groups=datum["conversation_id"])
    ]

    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category

    fold_cat_prod = fold_cat[datum.speaker == "Speaker1"]
    fold_cat_comp = fold_cat[datum.speaker != "Speaker1"]

    return (fold_cat_prod, fold_cat_comp)


def get_kfolds(X, fold_num=10):
    print("Using kfolds")
    skf = KFold(n_splits=fold_num, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(X.shape[0]))]
    fold_cat = np.zeros(X.shape[0])
    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category
    return fold_cat


def write_encoding_results(args, cor_results, elec_name, mode):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        cor_results: correlation results
        elec_name: electrode name as a substring of filename
        mode: 'prod' or 'comp'

    Returns:
        None
    """
    trial_str = append_jobid_to_string(args, mode)
    filename = os.path.join(
        args.full_output_dir, elec_name + trial_str + ".csv"
    )
    fold_filename = os.path.join(
        args.full_output_dir, elec_name + trial_str + "_fold.csv"
    )

    if len(cor_results) == 1: # no permutations
        cor_results = cor_results[0]

    # correlation for whole datum
    cor_datum = [cor_results[0]]
    with open(filename, "w") as csvfile:
        print("writing file")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(cor_datum)

    # correlation per fold
    cor_folds = cor_results[1:]
    df = pd.DataFrame(cor_folds)
    df.loc[len(df),:] = df.mean(axis=0)
    df.loc[len(df),:] = df.sem(axis=0,ddof=0)
    # FIXME I really don't like mean and sem here, maybe we do it in plotting instead?
    df.to_csv(fold_filename, index=False)

    return None


def append_jobid_to_string(args, speech_str):
    """Adds job id to the output eletrode.csv file.

    Args:
        args (Namespace): Contains all commandline agruments
        speech_str (string): Production (prod)/Comprehension (comp)

    Returns:
        string: concatenated string
    """
    speech_str = "_" + speech_str

    if args.job_id:
        trial_str = "_".join([speech_str, f"{args.job_id:02d}"])
    else:
        trial_str = speech_str

    return trial_str
