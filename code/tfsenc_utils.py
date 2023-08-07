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

from himalaya.ridge import GroupRidgeCV, RidgeCV
from himalaya.ridge import ColumnTransformerNoStack
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


def cv_leave_one_out(args, X, Y, col_idx):
    Xtra = X[X[:, col_idx] == 0, :]  # training on zeros
    Xtes = X[X[:, col_idx] == 1, :]  # testing on ones
    Ytra = Y[X[:, col_idx] == 0, :]
    YTES = Y[X[:, col_idx] == 1, :]

    if args.pca_to == 0:
        print("LM with no pca")
        model = make_pipeline(LinearRegression())
    else:
        print(f"PCA from {Xtra.shape[1]} to {args.pca_to}")
        model = make_pipeline(PCA(args.pca_to, whiten=True), LinearRegression())
    model.fit(Xtra, Ytra)
    YHAT = model.predict(Xtes)
    YHAT_NN = YHAT
    YHAT_NNT = YHAT

    return YHAT, YHAT_NN, YHAT_NNT, YTES


def cv_lm_003(
    args,
    X,
    Y,
    fold_num,
    given_fold=None,
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
        skf = KFold(n_splits=fold_num, shuffle=False)
        folds = [t[1] for t in skf.split(np.arange(nSamps))]

    YHAT = np.zeros((nSamps, nChans))
    YTES = np.zeros((nSamps, nChans))
    YHAT_NN = np.zeros((nSamps, nChans))
    YHAT_NNT = np.zeros((nSamps, nChans))

    # Go through each fold, and split
    for i in range(fold_num):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        folds_ixs = np.roll(range(fold_num), i)
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

        if "-br" in args.output_parent_dir:
            pass
            #     print("Banded Ridge")
            #     ct = ColumnTransformerNoStack(
            #         [
            #             ("cur_arb", StandardScaler(), slice(0, 50)),
            #             ("cur_pos", StandardScaler(), [50]),
            #             ("cur_stop", StandardScaler(), [51]),
            #             ("cur_shape", StandardScaler(), [52]),
            #             ("cur_prefix", StandardScaler(), [53]),
            #             ("cur_suffix", StandardScaler(), [54]),
            #         ]
            #     )
            # model = make_pipeline(
            #     ct,
            #     GroupRidgeCV(groups="input", cv=9),
            # )

        elif "-r" in args.output_parent_dir:
            pass
            # print("Ridge")
            # model = make_pipeline(StandardScaler(), RidgeCV())
        # Fit model
        elif args.pca_to == 0:
            print("LM with no pca")
            model = make_pipeline(LinearRegression())
        elif "9-3" in args.output_parent_dir:
            print(f"PCA from {Xtra.shape[1]} to {args.pca_to}")
            model = make_pipeline(
                StandardScaler(),
                PCA(args.pca_to, whiten=True),
                LinearRegression(),
            )
        elif "9-4" in args.output_parent_dir:
            print(f"PCA from {Xtra.shape[1]} to {args.pca_to}")
            model = make_pipeline(
                StandardScaler(),
                PCA(args.pca_to, whiten=True),
                LinearRegression(fit_intercept=False),
            )
        elif "9-5" in args.output_parent_dir:
            print(f"PCA from {Xtra.shape[1]} to {args.pca_to}")
            model = make_pipeline(
                PCA(args.pca_to, whiten=True),
                LinearRegression(),
            )

        model.fit(Xtra, Ytra)

        if "-br-" in args.output_parent_dir:
            if args.output_parent_dir.endswith("br-arb"):
                model[1].coef_[50:55, :] = 0
            elif args.output_parent_dir.endswith("br-pos"):
                model[1].coef_[0:50, :] = 0
                model[1].coef_[51:, :] = 0
            elif args.output_parent_dir.endswith("br-stop"):
                model[1].coef_[0:51, :] = 0
                model[1].coef_[52:, :] = 0
            elif args.output_parent_dir.endswith("br-shape"):
                model[1].coef_[0:52, :] = 0
                model[1].coef_[53:, :] = 0
            elif args.output_parent_dir.endswith("br-prefix"):
                model[1].coef_[0:53, :] = 0
                model[1].coef_[54, :] = 0
            elif args.output_parent_dir.endswith("br-suffix"):
                model[1].coef_[0:54, :] = 0

        # Predict
        foldYhat = model.predict(Xtes)
        YHAT[test_index, :] = foldYhat.reshape(-1, nChans)

        ###### regular near neighbor
        if "concat" not in args.output_parent_dir:
            # near neighbor
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtra)
            _, I = nbrs.kneighbors(Xtes)
            XtesNN = Xtra[I].squeeze()
            YHAT_NN[test_index, :] = model.predict(XtesNN)
            # YHAT_NN[test_index, :] = model.predict(Xtes) #HACK for is_stop

            # near neighbor test
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtes)
            _, I = nbrs.kneighbors()
            XtesNNT = Xtes[I].squeeze()
            YHAT_NNT[test_index, :] = model.predict(XtesNNT)
            # YHAT_NNT[test_index, :] = model.predict(Xtes) #HACK for is_stop

        ###### concat near neighbor (only NN on the last word embedding)
        else:
            if "glove" in args.output_parent_dir:
                last_idx = -50
            elif "symbolic-8" in args.output_parent_dir:
                last_idx = -125
            elif "symbolic-9" in args.output_parent_dir:
                last_idx = -75
            elif "symbolic-12" in args.output_parent_dir:
                last_idx = -50
            elif "symbolic-13" in args.output_parent_dir:
                last_idx = -11
            elif "symbolic-14" in args.output_parent_dir:
                last_idx = -1
            elif "symbolic-15" in args.output_parent_dir:
                last_idx = -16
            elif "symbolic-16" in args.output_parent_dir:
                last_idx = -19
            elif "symbolic-17" in args.output_parent_dir:
                last_idx = -28

            else:
                print("Wierd stuff")
                breakpoint()

            Xtran = Xtra[:, last_idx:]
            Xtesn = Xtes[:, last_idx:]
            Xtesc = Xtes[:, :last_idx]

            # near neighbor
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtran)
            _, I = nbrs.kneighbors(Xtesn)
            Xtesn = Xtran[I].squeeze()  # HACK comment out for is_stop
            XtesNN = np.hstack((Xtesc, Xtesn))
            YHAT_NN[test_index, :] = model.predict(XtesNN)

            # near neighbor test
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine")
            nbrs.fit(Xtesn)
            _, I = nbrs.kneighbors()
            Xtesn = Xtesn[I].squeeze()  # HACK comment out for is_stop
            XtesNNT = np.hstack((Xtesc, Xtesn))
            YHAT_NNT[test_index, :] = model.predict(XtesNNT)

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

    if "symbolic-9-0shot" in args.output_parent_dir:
        print("Deleting zero columns")
        X = X[:, (X.sum(axis=0) > 0).astype("bool")]  # deleting zero columns

        first_partial = args.output_parent_dir[
            args.output_parent_dir.find("symbolic-9-0shot") + 16 :
        ]
        if first_partial.find("concat") > 0:
            first_partial = first_partial[: first_partial.find("concat") - 1]
        col_idx = int(first_partial)
        assert col_idx < 0
        print(f"Taking index {col_idx}")
        PY_hat, PY_hat_nn, PY_hat_nnt, Ytes = cv_leave_one_out(
            args, X, Y, col_idx
        )
        rp = []
    else:
        PY_hat, PY_hat_nn, PY_hat_nnt, Ytes = cv_lm_003(args, X, Y, 10, fold)
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
    elif args.emb_type == "symbolic":
        stra = ""

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

    args.signal_file = "_".join(
        [str(args.sid), args.pkl_identifier, "signal.pkl"]
    )
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
