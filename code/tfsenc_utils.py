import csv
import os
import pickle
from functools import partial
from multiprocessing import Pool

import mat73
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy import stats
from scipy.spatial.distance import cdist

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


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


def cv_lm_003(
    X, Y, kfolds, near_neighbor=False, near_neighbor_test=True, given_fold=None
):
    """Cross-validated predictions from a regression model using sequential
        block partitions with nuisance regressors included in the training
        folds

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        kfolds ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Data size
    nSamps = X.shape[0]
    nChans = Y.shape[1] if Y.shape[1:] else 1

    # Extract only test folds
    if isinstance(given_fold, pd.Series):
        print("Provided Fold")
        folds = []
        given_fold = given_fold.tolist()
        for _ in set(given_fold):
            folds.append(np.array([], dtype=int))
        for index, fold in enumerate(given_fold):
            folds[int(fold)] = np.append(folds[int(fold)], index)
    else:
        print("Kfold")
        skf = KFold(n_splits=kfolds, shuffle=False)
        folds = [t[1] for t in skf.split(np.arange(nSamps))]

    YHAT = np.zeros((nSamps, nChans))
    YTES = np.zeros((nSamps, nChans))
    YHAT_NN = np.zeros((nSamps, nChans)) if near_neighbor else None
    YHAT_NNT = np.zeros((nSamps, nChans)) if near_neighbor_test else None

    # Go through each fold, and split
    for i in range(kfolds):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        folds_ixs = np.roll(range(kfolds), i)
        test_fold = folds_ixs[-1]
        train_folds = folds_ixs[:-1]

        test_index = folds[test_fold]
        train_index = np.concatenate([folds[j] for j in train_folds])

        # Extract each set out of the big matricies
        Xtra, Xtes = X[train_index], X[test_index]
        Ytra, Ytes = Y[train_index], Y[test_index]

        # yscaler = StandardScaler()
        # Ytra = yscaler.fit_transform(Ytra)
        # Ytes = yscaler.transform(Ytes)
        YTES[test_index, :] = Ytes

        # Fit model
        model = make_pipeline(PCA(50, whiten=True), LinearRegression())
        model.fit(Xtra, Ytra)

        # Predict
        foldYhat = model.predict(Xtes)
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

        ###### regular near neighbor
        if near_neighbor:
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtra)
            _, I = nbrs.kneighbors(Xtes)
            XtesNN = Xtra[I].squeeze()
            YHAT_NN[test_index, :] = model.predict(XtesNN)

        if near_neighbor_test:
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtes)
            _, I = nbrs.kneighbors()
            XtesNNT = Xtes[I].squeeze()
            YHAT_NNT[test_index, :] = model.predict(XtesNNT)

        ##### for glove concatenation (only nearestneighbors on the last 50d)
        # Xtran = Xtra[:, -50:]
        # Xtesn = Xtes[:, -50:]
        # Xtesc = Xtes[:, :-50]

        # if near_neighbor:
        #     nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
        #     nbrs.fit(Xtran)
        #     _, I = nbrs.kneighbors(Xtesn)
        #     Xtesn = Xtran[I].squeeze()
        #     XtesNN = np.hstack((Xtesc,Xtesn))
        #     YHAT_NN[test_index, :] = model.predict(XtesNN)

        # if near_neighbor_test:
        #     nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
        #     nbrs.fit(Xtesn)
        #     _, I = nbrs.kneighbors()
        #     Xtesn = Xtesn[I].squeeze()
        #     XtesNNT = np.hstack((Xtesc,Xtesn))
        #     YHAT_NNT[test_index, :] = model.predict(XtesNNT)

    return YHAT, YHAT_NN, YHAT_NNT, YTES


@jit(nopython=True)
def fit_model(X, y):
    """Calculate weight vector using normal form of regression.

    Returns:
        [type]: (X'X)^-1 * (X'y)
    """
    beta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
    return beta


@jit(nopython=True)
def build_Y(onsets, convo_onsets, convo_offsets, brain_signal, lags, window_size):
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

    Y1 = np.zeros((len(onsets), len(lags), 2 * half_window + 1))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)

        index_onsets = np.minimum(
            convo_offsets - half_window - 1,
            np.maximum(
                convo_onsets + half_window + 1,
                np.round_(onsets, 0, onsets) + lag_amount,
            ),
        )

        # subtracting 1 from starts to account for 0-indexing
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # vec = brain_signal[np.array(
        #     [np.arange(*item) for item in zip(starts, stops)])]

        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag, :] = brain_signal[start:stop].reshape(-1)

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


def encode_lags_numba(args, X, Y, fold):
    """[summary]
    Args:
        X ([type]): [description]
        Y ([type]): [description]
    Returns:
        [type]: [description]
    """
    if args.shuffle:
        np.random.shuffle(Y)

    Y = np.mean(Y, axis=-1)

    # First Differences Procedure
    # X = np.diff(X, axis=0)
    # Y = np.diff(Y, axis=0)

    PY_hat, PY_hat_nn, PY_hat_nnt, Ytes = cv_lm_003(
        X, Y, 10, args.test_near_neighbor, args.test_near_neighbor, fold
    )
    rp, _, _ = encColCorr(Y, PY_hat)

    if args.save_pred:
        fn = os.path.join(args.full_output_dir, args.current_elec + ".pkl")
        with open(fn, "wb") as f:
            pickle.dump(
                {
                    "electrode": args.current_elec,
                    "lags": args.lags,
                    "Y_signal": Ytes,
                    "Yhat_nn_signal": PY_hat_nn,
                    "Yhat_nnt_signal": PY_hat_nnt,
                    "Yhat_signal": PY_hat,
                },
                f,
            )

    return rp


def encoding_mp(_, args, prod_X, prod_Y, fold):
    perm_rc = encode_lags_numba(args, prod_X, prod_Y, fold)
    return perm_rc


def run_save_permutation_pr(args, prod_X, prod_Y, filename):
    """[summary]
    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        perm_rc = encode_lags_numba(args, prod_X, prod_Y)
    else:
        perm_rc = None

    return perm_rc


def run_save_permutation(args, prod_X, prod_Y, filename, fold=None):
    """[summary]

    Args:
        args ([type]): [description]
        prod_X ([type]): [description]
        prod_Y ([type]): [description]
        filename ([type]): [description]
    """
    if prod_X.shape[0]:
        if args.parallel:
            print(f"Running {args.npermutations} in parallel")
            with Pool(16) as pool:
                perm_prod = pool.map(
                    partial(
                        encoding_mp,
                        args=args,
                        prod_X=prod_X,
                        prod_Y=prod_Y,
                        fold=fold,
                    ),
                    range(args.npermutations),
                )
        else:
            perm_prod = []
            for i in range(args.npermutations):
                perm_prod.append(encoding_mp(i, args, prod_X, prod_Y, fold))
                # print(max(perm_prod[-1]), np.mean(perm_prod[-1]))

        with open(filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(perm_prod)


def load_header(conversation_dir, subject_id):
    """[summary]

    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID

    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, subject_id, "misc")
    header_file = os.path.join(misc_dir, subject_id + "_header.mat")
    if not os.path.exists(header_file):
        print(f"[WARN] no header found in {misc_dir}")
        return
    header = mat73.loadmat(header_file)
    # labels = header.header.label
    labels = header["header"]["label"]

    return labels


def create_output_directory(args):
    # output_prefix_add = '-'.join(args.emb_file.split('_')[:-1])

    # folder_name = folder_name + '-pca_' + str(args.reduce_to) + 'd'
    # full_output_dir = os.path.join(args.output_dir, folder_name)

    folder_name = "-".join([args.output_prefix, str(args.sid)])
    folder_name = folder_name.strip("-")
    full_output_dir = os.path.join(
        os.getcwd(),
        "results",
        args.project_id,
        args.output_parent_dir,
        folder_name,
    )

    os.makedirs(full_output_dir, exist_ok=True)

    return full_output_dir


def encoding_regression_pr(args, datum, elec_signal, name):
    """[summary]
    Args:
        args (Namespace): Command-line inputs and other configuration
        sid (str): Subject ID
        datum (DataFrame): ['word', 'onset', 'offset', 'speaker', 'accuracy']
        elec_signal (numpy.ndarray): of shape (num_samples, 1)
        name (str): electrode name
    """

    datum = datum[datum.adjusted_onset.notna()]

    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == "Speaker1", :]
    comp_X = X[datum.speaker != "Speaker1", :]

    prod_Y = Y[datum.speaker == "Speaker1", :]
    comp_Y = Y[datum.speaker != "Speaker1", :]

    # Run permutation and save results
    prod_corr = run_save_permutation_pr(args, prod_X, prod_Y, None)
    comp_corr = run_save_permutation_pr(args, comp_X, comp_Y, None)

    return (prod_corr, comp_corr)


def encoding_regression(args, datum, elec_signal, name):

    output_dir = args.full_output_dir
    datum = datum[datum.adjusted_onset.notna()]

    # Build design matrices
    X, Y = build_XY(args, datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[datum.speaker == "Speaker1", :]
    comp_X = X[datum.speaker != "Speaker1", :]

    prod_Y = Y[datum.speaker == "Speaker1", :]
    comp_Y = Y[datum.speaker != "Speaker1", :]

    print(f"{args.sid} {name} Prod: {len(prod_X)} Comp: {len(comp_X)}")
    args.current_elec = name

    # Run permutation and save results
    # trial_str = append_jobid_to_string(args, "prod")
    # filename = os.path.join(output_dir, name + trial_str + ".csv")
    # run_save_permutation(args, prod_X, prod_Y, filename)

    trial_str = append_jobid_to_string(args, "comp")
    filename = os.path.join(output_dir, name + trial_str + ".csv")

    if "fold" in datum.columns:
        fold = datum.fold
    else:
        fold = None
    run_save_permutation(args, comp_X, comp_Y, filename, fold)

    return


def setup_environ(args):
    """Update args with project specific directories and other flags"""
    PICKLE_DIR = os.path.join(
        os.getcwd(), "data", args.project_id, str(args.sid), "pickles"
    )
    path_dict = dict(PICKLE_DIR=PICKLE_DIR)

    stra = "cnxt_" + str(args.context_length)
    if args.emb_type == "glove50":
        stra = ""
        args.layer_idx = 1

    # TODO make an arg
    zeroshot = ""
    # zeroshot = '_0shot'  #
    # zeroshot = '_0shot_new'  #
    # zeroshot = '_0shot_bbd'  # bobbi's datum

    args.emb_file = "_".join(
        [
            str(args.sid),
            args.pkl_identifier,
            args.emb_type,
            stra,
            f"layer_{args.layer_idx:02d}",
            f"embeddings{zeroshot}.pkl",
        ]
    )
    args.load_emb_file = args.emb_file.replace("__", "_")

    args.signal_file = "_".join([str(args.sid), args.pkl_identifier, "signal.pkl"])
    args.electrode_file = "_".join([str(args.sid), "electrode_names.pkl"])
    args.stitch_file = "_".join(
        [str(args.sid), args.pkl_identifier, "stitch_index.pkl"]
    )

    args.output_dir = os.path.join(os.getcwd(), "results")
    args.full_output_dir = create_output_directory(args)

    vars(args).update(path_dict)
    return args


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
