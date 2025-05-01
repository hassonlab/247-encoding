import csv
import os
import numpy as np
import pandas as pd
import pickle
import torch
from numba import jit, prange
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from himalaya.ridge import GroupRidgeCV, RidgeCV, ColumnTransformerNoStack
from himalaya.kernel_ridge import (
    MultipleKernelRidgeCV,
    KernelRidgeCV,
    ColumnKernelizer,
    Kernelizer,
)
from himalaya.scoring import correlation_score, correlation_score_split

do_debug = False

# @jit(nopython=True)
def build_Y(brain_signal, onsets, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    half_window = round((window_size / 1000) * 512 / 2)  # convert to signal fs
    Y1 = np.zeros((len(onsets), len(lags)))
    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)  # convert fo signal fs
        index_onsets = np.round_(onsets, 0, onsets) + lag_amount  # lag onsets
        starts = index_onsets - half_window - 1  # lag window onset
        stops = index_onsets + half_window  # lag window offset
        starts = np.array(starts, dtype=int)
        stops = np.array(stops, dtype=int)
        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag] = np.mean(
                brain_signal[start:stop].reshape(-1)
            )  # average lag window
    return Y1


def get_groupkfolds(datum, X, Y, fold_num=10):
    fold_cat = np.zeros(datum.shape[0])
    grpkfold = GroupKFold(n_splits=fold_num)
    folds = [t[1] for t in grpkfold.split(X, Y, groups=datum["conversation_id"])]

    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category

    fold_cat_comp = fold_cat[datum.speaker != "Speaker1"]
    fold_cat_prod = fold_cat[datum.speaker == "Speaker1"]

    return (fold_cat_comp, fold_cat_prod)


def get_kfolds(X, fold_num=10):
    print("Using kfolds")
    skf = KFold(n_splits=fold_num, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(X.shape[0]))]
    fold_cat = np.zeros(X.shape[0])
    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category
    return fold_cat


def encoding_setup(args, elec_name, elec_datum, elec_signal):
    # Build x and y matrices
    #elec_datum.to_csv('/scratch/gpfs/cc27/elec_datum_20250127_refrecur.csv')
    #np.savez('/scratch/gpfs/cc27/elec_signal_recrecur.npz', elec_signal=elec_signal, lags=args.lags, window_size=args.window_size)
    X = np.stack(elec_datum.embeddings).astype("float64")
    Y = build_Y(
        elec_signal.reshape(-1, 1),
        elec_datum.adjusted_onset.values,
        np.array(args.lags),
        args.window_size,
    )
    X = X.astype("float32")
    Y = Y.astype("float32")

    extra_train_comp_data = None
    extra_train_prod_data = None
    extra_test_comp_data = None
    extra_test_prod_data = None
    if "data_subset_type" in args:
        if args.data_subset_type == "ref_recap_test_only":
            ref_recap_indices = elec_datum.annot_type.isin(["ref", "recap"])
            not_ref_recap_indices = ~ref_recap_indices
            
            # Create extra test comp/prod
            extra_test_X_comp_data = X[
                 (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                 & (elec_datum.speaker != "Speaker1"),
                 :,
             ]
            extra_test_Y_comp_data = Y[
                 (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                 & (elec_datum.speaker != "Speaker1"),
                 :,
             ]
            extra_test_X_prod_data = X[
                 (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                 & (elec_datum.speaker == "Speaker1"),
                 :,
             ]
            extra_test_Y_prod_data = Y[
                 (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                 & (elec_datum.speaker == "Speaker1"),
                 :,
             ]
            extra_test_comp_data = extra_test_X_comp_data, extra_test_Y_comp_data
            extra_test_prod_data = extra_test_X_prod_data, extra_test_Y_prod_data

            # save test elec datum
            extra_test_comp_elec_datum = elec_datum[
                (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                & (elec_datum.speaker != "Speaker1")
            ]
            extra_test_prod_elec_datum = elec_datum[
                (elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                & (elec_datum.speaker == "Speaker1")
            ]
            if not os.path.exists(os.path.join(args.output_dir, "prod_datum_extra_test.csv")):
                extra_test_comp_elec_datum.to_csv(
                    os.path.join(args.output_dir, "comp_datum_extra_test.csv"), index=False
                )
                extra_test_prod_elec_datum.to_csv(
                    os.path.join(args.output_dir, "prod_datum_extra_test.csv"), index=False
                )
            
        elif args.data_subset_type == "ref_only":
            print("before subsetting", X.shape, Y.shape)
            # Update X/Y/elec_datum
            X = X[elec_datum.annot_type.isin(["ref"]), :]
            Y = Y[elec_datum.annot_type.isin(["ref"]), :]
            elec_datum = elec_datum[
                elec_datum.annot_type.isin(["ref"])
            ]
        elif args.data_subset_type == "ref_recap_only":
            print("before subsetting", X.shape, Y.shape)
            if "use_nonannot_as_train" in args and args.use_nonannot_as_train:
                extra_train_X_comp_data = X[
                    (~elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                    & (elec_datum.speaker != "Speaker1"),
                    :,
                ]
                extra_train_Y_comp_data = Y[
                    (~elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                    & (elec_datum.speaker != "Speaker1"),
                    :,
                ]
                extra_train_X_prod_data = X[
                    (~elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                    & (elec_datum.speaker == "Speaker1"),
                    :,
                ]
                extra_train_Y_prod_data = Y[
                    (~elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
                    & (elec_datum.speaker == "Speaker1"),
                    :,
                ]
                extra_train_comp_data = extra_train_X_comp_data, extra_train_Y_comp_data
                extra_train_prod_data = extra_train_X_prod_data, extra_train_Y_prod_data
            X = X[elec_datum.annot_type.isin(["ref", "recap", "mapping"]), :]
            Y = Y[elec_datum.annot_type.isin(["ref", "recap", "mapping"]), :]
            elec_datum = elec_datum[
                elec_datum.annot_type.isin(["ref", "recap", "mapping"])
            ]
        elif args.data_subset_type == "first_match_ref_recap_only":
            subset_length = sum(elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
            X = X[:subset_length, :]
            Y = Y[:subset_length, :]
            elec_datum = elec_datum.iloc[:subset_length]
        elif args.data_subset_type == "random_match_ref_recap_only":
            subset_length = sum(elec_datum.annot_type.isin(["ref", "recap", "mapping"]))
            indices = np.random.choice(np.arange(X.shape[0]), subset_length, replace=False)
            X = X[indices, :]
            Y = Y[indices, :]
            elec_datum = elec_datum.iloc[:subset_length]
        print("after subsetting", X.shape, Y.shape)
    # Split into production and comprehension
    prod_X = X[elec_datum.speaker == "Speaker1", :]
    comp_X = X[elec_datum.speaker != "Speaker1", :]
    prod_Y = Y[elec_datum.speaker == "Speaker1", :]
    comp_Y = Y[elec_datum.speaker != "Speaker1", :]

    # Print shapes
    print(f"Prod X: {prod_X.shape}, Prod Y: {prod_Y.shape}")
    print(f"Comp X: {comp_X.shape}, Comp Y: {comp_Y.shape}")
    elec_datum_prod = elec_datum[elec_datum.speaker == "Speaker1"]
    elec_datum_comp = elec_datum[elec_datum.speaker != "Speaker1"]
    if not os.path.exists(os.path.join(args.output_dir, "prod_datum.csv")):
        # Just save once, because same for all elecs
        elec_datum_prod.to_csv(
            os.path.join(args.output_dir, "prod_datum.csv"), index=False
        )
        elec_datum_comp.to_csv(
            os.path.join(args.output_dir, "comp_datum.csv"), index=False
        )

    # get folds
    if args.project_id == "podcast":  # podcast
        fold_cat_comp = get_kfolds(comp_X, args.cv_fold_num)
        fold_cat_prod = []
    elif len(args.conv_ids) < args.cv_fold_num * 2:  # small num of convos, also 798
        print(
            f"{args.sid} {elec_name} has less convos than fold nums, doing kfold instead"
        )
        fold_cat_comp = get_kfolds(comp_X, args.cv_fold_num)
        fold_cat_prod = get_kfolds(prod_X, args.cv_fold_num)
    else:  # Get groupkfolds
        fold_cat_comp, fold_cat_prod = get_groupkfolds(
            elec_datum, X, Y, args.cv_fold_num
        )

    # Oragnize
    comp_data = comp_X, comp_Y, fold_cat_comp
    prod_data = prod_X, prod_Y, fold_cat_prod

    return comp_data, prod_data, extra_train_comp_data, extra_train_prod_data, extra_test_comp_data, extra_test_prod_data


def encoding_correlation(CA, CB):
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


def encoding_regression_permutation(args, X, Y, folds, num_perm=1000, min_roll=500):
    """Run regression for VM

    Args:
        args (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        folds (_type_): _description_
        extra_train_data (_type_, optional): Extra training X and Y (never used in val split). Defaults to None.
        extra_test_data (_type_, optional): Extra testing X and Y (never used in train split). Defaults to None.

    Returns:
        _type_: _description_
    """
    nSamps = X.shape[0]
    nChans = Y.shape[1] if Y.shape[1:] else 1

    YHAT = np.zeros((nSamps, nChans * num_perm)).astype("float32") * np.nan
    Ynew = np.zeros((nSamps, nChans * num_perm)).astype("float32") * np.nan
    YHAT_extra = None
    Ynew_extra = None
    corrs = []
    corrs_split = []
    
    #Y = np.nan_to_num(Y)
    # Circular shift
    np.random.seed(123)
    Yperm = np.zeros((nSamps, nChans * num_perm))
    for i in range(num_perm):
        Yperm[:, i * nChans : (i + 1) * nChans] = np.roll(Y, np.random.choice(range(min_roll, nSamps - min_roll)), axis=0)

    alphas = np.array([10581255000000.0])
    solver_params = {"n_targets_batch": 161 * num_perm, "alphas": alphas, "n_iter": 1}
    if getattr(args, "kernel_sizes", None) is not None:
        kernel_sizes_cumsum = np.cumsum(args.kernel_sizes)
        if getattr(args, "bridge_type", None) == "GroupRidgeCV":
            print("Using GroupRidgeCV")
            ct = ColumnTransformerNoStack([(f"group_{i}", StandardScaler(), np.arange(kernel_start, kernel_stop))
                                            for i, (kernel_start, kernel_stop) in enumerate(zip([0]+list(kernel_sizes_cumsum[:-1]), kernel_sizes_cumsum))])
            grcv_model = GroupRidgeCV(groups="input", solver_params=solver_params)
            model = make_pipeline(ct, grcv_model)
        else:
            print("Using MultipleKernelRidgeCV")
            ck = ColumnKernelizer([(f"kernel_{i}", Kernelizer(kernel="linear"), np.arange(kernel_start, kernel_stop))
                                for i, (kernel_start, kernel_stop) in enumerate(zip([0]+list(kernel_sizes_cumsum[:-1]), kernel_sizes_cumsum))])
            model = make_pipeline(StandardScaler(), ck, MultipleKernelRidgeCV(kernels="precomputed", solver_params=solver_params))
    else:
        print(f"Running RidgeCV")
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, solver_params=solver_params),
        )
    
    for i in range(0, args.cv_fold_num):
        Xtrain, Xtest = X[folds != i], X[folds == i]
        Ytrain, Ytest = Yperm[folds != i], Yperm[folds == i]
        
        # Skip rows where there are nan X vals. (Might have nan Xs if e.g., retrieval CL0 with no retrieved words.)
        non_nan_rows_train = np.where(~np.isnan(Xtrain.sum(1)))
        Xtrain = Xtrain[non_nan_rows_train]
        Ytrain = Ytrain[non_nan_rows_train]
        non_nan_rows_test = np.where(~np.isnan(Xtest.sum(1)))
        Xtest = Xtest[non_nan_rows_test]
        Ytest = Ytest[non_nan_rows_test]
        
        import time
        t0 = time.time()
        model.fit(Xtrain, Ytrain)
        print(time.time() - t0)
        foldYhat = model.predict(Xtest)
        fold_corrs = correlation_score(Ytest, foldYhat).cpu()
        corrs.append(fold_corrs)
        if getattr(args, "bridge_type", None) != "RidgeCV":
            foldYhat_split = model.predict(Xtest, split=True)
            fold_cor_split = correlation_score_split(Ytest, foldYhat_split).cpu()
        else:
            fold_cor_split = None
        corrs_split.append(fold_cor_split)
        
        Ynew[folds == i, :][non_nan_rows_test] = Ytest.reshape(-1, nChans * num_perm)
        YHAT[folds == i, :][non_nan_rows_test] = foldYhat.reshape(-1, nChans * num_perm).cpu()
    return (YHAT, Ynew, corrs, corrs_split, YHAT_extra, Ynew_extra)


def encoding_regression(args, X, Y, folds, extra_train_data=None, extra_test_data=None,
                        n_alphas_batch=50, n_iter=25):
    """Run regression for VM

    Args:
        args (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        folds (_type_): _description_
        extra_train_data (_type_, optional): Extra training X and Y (never used in val split). Defaults to None.
        extra_test_data (_type_, optional): Extra testing X and Y (never used in train split). Defaults to None.

    Returns:
        _type_: _description_
    """
    nSamps = X.shape[0]
    nChans = Y.shape[1] if Y.shape[1:] else 1

    YHAT = np.zeros((nSamps, nChans)).astype("float32")
    Ynew = np.zeros((nSamps, nChans)).astype("float32")
    YHAT_extra = None
    Ynew_extra = None
    corrs = []
    corrs_split = []
    params = {"alphas": [], "gammas": []}
    
    if extra_test_data:
        YHAT_extra = np.zeros((args.cv_fold_num, extra_test_data[0].shape[0], nChans)).astype("float32")
        Ynew_extra = np.zeros((args.cv_fold_num, extra_test_data[0].shape[0], nChans)).astype("float32")

    # TODO: TAKE OUT FIGURE OUT WHAT IS HAPPENING.
    Y = np.nan_to_num(Y)
    if do_debug:
        n_iter = 1
    elif getattr(args, "n_iter", None) is not None:
        n_iter = args.n_iter
    
    all_fold_yhat_split = []
    for i in range(0, args.cv_fold_num):

        Xtrain, Xtest = X[folds != i], X[folds == i]
        Ytrain, Ytest = Y[folds != i], Y[folds == i]
        if extra_train_data is not None:
            Xtrain = np.vstack([Xtrain, extra_train_data[0]])
            Ytrain = np.vstack([Ytrain, extra_train_data[1]])
        if extra_test_data is not None:
            Xtest_extra = extra_test_data[0]
            Ytest_extra = extra_test_data[1] - np.mean(Ytrain, axis=0)

        # Skip rows where there are nan X vals. (Might have nan Xs if e.g., retrieval CL0 with no retrieved words.)
        non_nan_rows_train = np.where(~np.isnan(Xtrain.sum(1)))
        Xtrain = Xtrain[non_nan_rows_train]
        Ytrain = Ytrain[non_nan_rows_train]
        non_nan_rows_test = np.where(~np.isnan(Xtest.sum(1)))
        Xtest = Xtest[non_nan_rows_test]
        Ytest = Ytest[non_nan_rows_test]

        alphas = np.logspace(0, 20, 10)
        if n_iter == 1:
            alphas = np.array([4641591])
            concentration = 1e9
        else:
            concentration = [0.1, 1.0]
 
        if getattr(args, "kernel_sizes", None) is not None:
            kernel_sizes_cumsum = np.cumsum(args.kernel_sizes)
            if getattr(args, "bridge_type", None) == "GroupRidgeCV":
                print("Using GroupRidgeCV")
                ct = ColumnTransformerNoStack([(f"group_{i}", StandardScaler(), np.arange(kernel_start, kernel_stop))
                                               for i, (kernel_start, kernel_stop) in enumerate(zip([0]+list(kernel_sizes_cumsum[:-1]), kernel_sizes_cumsum))])
                grcv_model = GroupRidgeCV(groups="input", solver_params=dict(alphas=alphas, n_iter=n_iter, n_alphas_batch=n_alphas_batch))
                model = make_pipeline(ct, grcv_model)
            elif getattr(args, "bridge_type", None) == "RidgeCV":
                print(f"Running RidgeCV, emb_dim = {Xtrain.shape[1]}")
                solver_params = {"n_alphas_batch": n_alphas_batch}
                model = make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=alphas, solver_params=solver_params),
                )
            else:
                print("Using MultipleKernelRidgeCV")
                ck = ColumnKernelizer([(f"kernel_{i}", Kernelizer(kernel="linear"), np.arange(kernel_start, kernel_stop))
                                    for i, (kernel_start, kernel_stop) in enumerate(zip([0]+list(kernel_sizes_cumsum[:-1]), kernel_sizes_cumsum))])
                model = make_pipeline(StandardScaler(), ck, MultipleKernelRidgeCV(kernels="precomputed", solver_params=dict(alphas=alphas, n_iter=n_iter, n_alphas_batch=n_alphas_batch, concentration=concentration)))
        elif not args.ridge:  # ols
            if args.pca_to == 0:
                print(f"Running OLS, emb_dim = {Xtrain.shape[1]}")
                if args.himalaya:
                    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-9]))
                else:
                    model = make_pipeline(StandardScaler(), LinearRegression())
            else:  # pca + ols
                print(f"Running PCA (from {Xtest.shape[1]} to {args.pca_to}) + OLS")
                model = make_pipeline(
                    StandardScaler(), PCA(args.pca_to, whiten=True), LinearRegression()
                )
        else:  # ridge cv
            solver_params = {"n_alphas_batch": 5}
            if Xtrain.shape[0] < Xtrain.shape[1]:
                print(f"Running KernelRidgeCV, emb_dim = {Xtrain.shape[1]}")
                model = make_pipeline(StandardScaler(), KernelRidgeCV(alphas=alphas))
            else:
                print(f"Running RidgeCV, emb_dim = {Xtrain.shape[1]}")
                model = make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=alphas, solver_params=solver_params),
                )
        torch.cuda.empty_cache()
        model.fit(Xtrain, Ytrain)
        alphas = model.steps[2][1].best_alphas_
        gammas = np.exp(model.steps[2][1].deltas_.cpu()) * alphas.cpu()
        if getattr(args, "save_params", False):
            params["alphas"].append(alphas.cpu().numpy())
            params["gammas"].append(gammas.cpu().numpy())
        # Prediction & Correlation
        foldYhat = model.predict(Xtest)
        fold_cors = correlation_score(Ytest, foldYhat)
        # Save split scores
        if getattr(args, "bridge_type", None) != "RidgeCV":
            foldYhat_split = model.predict(Xtest, split=True)
            fold_cor_split = correlation_score_split(Ytest, foldYhat_split)
            all_fold_yhat_split.append(foldYhat_split.cpu()) 
        else:
            fold_cor_split = None
        
        corrs.append(fold_cors)
        corrs_split.append(fold_cor_split)
        if extra_test_data:
            foldYhat_extra = model.predict(Xtest_extra)
            YHAT_extra[i, :] = foldYhat_extra.reshape(-1, nChans)
            Ynew_extra[i, :] = Ytest_extra.reshape(-1, nChans)
        if torch.is_tensor(Ytest):
            Ytest = Ytest.cpu()
        if torch.is_tensor(foldYhat):
            foldYhat = foldYhat.cpu()
        indices_to_update = np.where(folds == i)[0][non_nan_rows_test]
        Ynew[indices_to_update, :] = Ytest.reshape(-1, nChans)
        YHAT[indices_to_update, :] = foldYhat.reshape(-1, nChans)

    return (YHAT, Ynew, corrs, corrs_split, YHAT_extra, Ynew_extra, all_fold_yhat_split, params)

#@profile
def run_encoding(args, X, Y, folds, extra_train_data=None, extra_test_data=None, permute=False):

    # train lm and predict
    if permute:
        all_fold_yhat_split = None
        params = None
        Y_hat, Y_new, corrs, corrs_split, Y_hat_extra, Y_new_extra = encoding_regression_permutation(args, X, Y, folds)
    else:
        Y_hat, Y_new, corrs, corrs_split, Y_hat_extra, Y_new_extra, all_fold_yhat_split, params = encoding_regression(args, X, Y, folds, extra_train_data, extra_test_data)

    # # Old correlation
    # rps = []
    # # correlation for whole datum
    # rp, _, _ = encoding_correlation(Y_new, Y_hat)
    # rps.append(rp)

    # # correlation per folds
    # for i in np.unique(folds).astype(int):
    #     rp_fold, _, _ = encoding_correlation(Y_new[folds == i], Y_hat[folds == i])
    #     rps.append(rp_fold)

    # New correlation
    corr_datum = correlation_score(Y_new, Y_hat)
    if torch.is_tensor(corr_datum):  # torch tensor
        corrs.append(corr_datum)
        corrs = torch.stack([corr.cpu() for corr in corrs])
    else:
        corrs.append(corr_datum)
        corrs = np.stack(corrs)
    if corrs_split is not None:
        if torch.is_tensor(corrs_split[0]):
            corrs_split = torch.stack(corrs_split)
        else:
            corrs_split = np.stack(corrs_split)

    return corrs, corrs_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, all_fold_yhat_split, params

#@profile
def write_encoding_results(args, results, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, filename, folds=None, all_fold_yhat_split=None,
                           params=None):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        results: correlation results
        filename: usually electrode name plus 'prod' or 'comp'

    Returns:
        None
    """
    filename = os.path.join(args.output_dir, filename)
    print(f"Writing to {filename}")
    if torch.is_tensor(results):
        results = results.cpu().numpy()
    if torch.is_tensor(result_split):
        result_split = result_split.cpu().numpy()
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False, header=False)
    np.savez(filename.replace(".csv", "_split.npz"), result_split=result_split)
    
    if do_debug:
        import pdb; pdb.set_trace()
        
    if "save_preds" in args and args.save_preds:
        if all_fold_yhat_split is not None:
            if do_debug:
                import pdb; pdb.set_trace()
            with open(filename.replace(".csv", "_yhat_split.p"), "wb") as f:
                pickle.dump({"all_fold_yhat_split": all_fold_yhat_split, "Y_new": Y_new}, f)
                
        if torch.is_tensor(Y_hat):
            Y_hat = Y_hat.cpu().numpy()
        if torch.is_tensor(Y_new):
            Y_new = Y_new.cpu().numpy()
        if torch.is_tensor(folds):
            folds = folds.cpu().numpy()
        np.savez(
            filename.replace(".csv", ".npz"), Y_hat=Y_hat, Y_new=Y_new, folds=folds,
            Y_hat_extra=Y_hat_extra, Y_new_extra=Y_new_extra,
        )
    if params is not None:
        np.savez(
            filename.replace(".csv", "_params.npz"),
            alphas=params["alphas"],
            gammas=params["gammas"],
        )

    return
