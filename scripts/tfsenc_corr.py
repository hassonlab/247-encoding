import csv
import glob
import os
import yaml
import argparse
import getpass
import subprocess
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score, correlation_score_split
from utils import load_pickle, main_timer


FEAT_SPACES = ["llama", "wordlen", "gaplen", "all"]


def get_git_hash() -> str:
    """Get git hash as string"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def clean_lm_model_name(item):
    """Remove unnecessary parts from the language model name.

    Args:
        item (str/list): full model name from HF Hub

    Returns:
        (str/list): pretty model name

    Example:
        clean_lm_model_name(EleutherAI/gpt-neo-1.3B) == 'gpt-neo-1.3B'
    """
    if isinstance(item, str):
        return item.split("/")[-1]

    if isinstance(item, list):
        return [clean_lm_model_name(i) for i in item]

    print("Invalid input. Please check.")


def parse_arguments():
    """Read arguments from yaml config file

    Returns:
        namespace: all arguments from yaml config file
    """
    # parse yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", nargs="*", type=str, default="[config.yml]")
    parser.add_argument("--elecs", nargs="*", type=int, default=0)
    args = parser.parse_args()

    all_yml_args = {}
    for config_file in args.config_file:
        with open(config_file, "r") as file:
            yml_args = yaml.safe_load(file)
            all_yml_args = all_yml_args | yml_args

    # get username
    user_id = getpass.getuser()
    all_yml_args["user_id"] = user_id
    all_yml_args["git_hash"] = get_git_hash()

    # clean up args
    elecs = args.elecs
    args = argparse.Namespace(**all_yml_args)
    try:  # eval lists
        args.elecs = eval(args.elecs)
        args.conv_ids = eval(args.conv_ids)
        args.lags = eval(args.lags)
    except:
        print("List parameter failed to eval")

    if isinstance(elecs, list):
        print(f"{elecs=}")
        # args.elecs = elecs  # replace config elecs with elecs from Makefile
        print("Running for 30 elecs")
        args.elecs = [
            elecs[0] + i for i in np.arange(0, 30)
        ]  # replace config elecs with elecs from Makefile

    if args.emb == "glove50":  # for glove, fix layer and context len
        args.layer_idx = 0
        args.context_length = 1
    else:
        args.emb = clean_lm_model_name(args.emb)

    return args, all_yml_args


def process_electrodes(args):
    """Process electrodes for subjects (requires electrode list or sig elec file)

    Args:
        args (namespace): commandline arguments

    Returns:
        electrode_info (dict): each item in the format (sid, elec_id): elec_name
    """
    ds = load_pickle(args.electrode_file_path)
    df = pd.DataFrame(ds)
    if args.sig_elec_file is not None:  # sig elec files
        sig_elec_list = pd.read_csv(args.sig_elec_file_path).rename(
            columns={"electrode": "electrode_name"}
        )
        df["subject"] = df.subject.astype("int64")
        sid_sig_elec_list = pd.merge(
            df, sig_elec_list, how="inner", on=["subject", "electrode_name"]
        )
        assert len(sig_elec_list) == len(sid_sig_elec_list), "Sig Elecs Missing"
        electrode_info = {
            (values["subject"], values["electrode_id"]): values["electrode_name"]
            for _, values in sid_sig_elec_list.iterrows()
        }

    else:  # electrode list for 1 sid
        assert len(args.elecs) > 0, "Need electrode list since no sig_elec_list"
        electrode_info = {
            (args.sid, key): next(
                iter(
                    df.loc[
                        (df.subject == str(args.sid)) & (df.electrode_id == key),
                        "electrode_name",
                    ]
                ),
                None,
            )
            for key in args.elecs
        }

    return electrode_info


def setup_environ(args):
    """
    Update args with project specific directories and other flags

    Args:
        args (namespace): arguments

    Returns:
        args (namespace): arguments plus directory paths
    """

    # input directory paths (pickles)
    DATA_DIR = os.path.join(os.getcwd(), "data")
    PICKLE_DIR = os.path.join(DATA_DIR, args.project_id, str(args.sid), "pickles")

    args.electrode_file_path = os.path.join(
        PICKLE_DIR, ("_".join([str(args.sid), "electrode_names.pkl"]))
    )

    # output directory paths
    OUTPUT_DIR = os.path.join(os.getcwd(), "results", args.project_id)
    RESULT_PARENT_DIR = f"kw-tfs-{args.sid}-{args.output_dir_name}-%s"
    RESULT_CHILD_DIR = "kw-200ms"
    args.output_dir = os.path.join(OUTPUT_DIR, RESULT_PARENT_DIR, RESULT_CHILD_DIR)
    for feat in FEAT_SPACES:
        os.makedirs(args.output_dir % feat, exist_ok=True)

    if torch.cuda.is_available():
        print("set backend to cuda")
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        print("set backend to cpu numpy")
        backend = set_backend("numpy", on_error="warn")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # HACK

    return args


################################
########### TEST COR ###########
################################


def correlation_score_split_folds(df, ynew, yhat):
    """
    Calculate correlation score for each fold in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing 'folds' column.
        ynew (np.ndarray): New predictions.
        yhat (np.ndarray): Predicted values.

    Returns:
        cor_res (torch.Tensor): Correlation results averaged across folds
    """
    cor_res = []
    for fold in df.folds.unique():
        fold_df = df[df.folds == fold]
        cor_res_fold = correlation_score_split(
            ynew[df.index.get_indexer(fold_df.index)],
            yhat[:, df.index.get_indexer(fold_df.index), :],
        )
        cor_res.append(cor_res_fold)
    cor_res = torch.stack(cor_res, axis=0).mean(axis=0)
    return cor_res


def filter_cor(args, df, yhat, ynew):
    """
    Filter the datum and then calculate correlation based on the test mode specified in args.
    Args:
        args (namespace): Arguments containing test mode.
        df (pd.DataFrame): DataFrame containing the data.
        yhat (np.ndarray): Predicted values.
        ynew (np.ndarray): New predictions.
    Returns:
        cor_res (torch.Tensor): Correlation results after filtering.
    """
    assert len(df) == yhat.shape[1] == ynew.shape[0]

    if "filter-no-start" in args.test_mod:
        thresh = 2
        yhat = yhat[:, df.onset >= df.utt_onset + thresh * 512, :]
        ynew = ynew[df.onset >= df.utt_onset + thresh * 512, :]
        cor_res = correlation_score_split(ynew, yhat)
    elif "filter-short" in args.test_mod:
        thresh = 2
        yhat = yhat[:, df.utt_offset <= df.utt_onset + thresh * 512, :]
        ynew = ynew[df.utt_offset <= df.utt_onset + thresh * 512, :]
        cor_res = correlation_score_split(ynew, yhat)
    elif "filter-long" in args.test_mod:
        thresh = 2
        yhat = yhat[:, df.utt_offset >= df.utt_onset + thresh * 512, :]
        ynew = ynew[df.utt_offset >= df.utt_onset + thresh * 512, :]
        cor_res = correlation_score_split(ynew, yhat)
    elif "align-utt-2" in args.test_mod:  # align to lag 2s
        in_utt = df.utt_offset_prev - df.onset <= 2 * 512
        in_utt &= df.utt_onset_next - df.onset >= 2 * 512
        yhat = yhat[:, in_utt, :]
        ynew = ynew[in_utt, :]
        df = df[in_utt].copy()
        cor_res = correlation_score_split_folds(df, ynew, yhat)
    elif "align-utt-5" in args.test_mod:  # align to lag 5s
        in_utt = df.utt_offset_prev - df.onset <= 5 * 512
        in_utt &= df.utt_onset_next - df.onset >= 5 * 512
        yhat = yhat[:, in_utt, :]
        ynew = ynew[in_utt, :]
        df = df[in_utt].copy()
        cor_res = correlation_score_split_folds(df, ynew, yhat)
    elif "align-utt" in args.test_mod:  # cut off lags outside of utt
        assert len(args.lags) == yhat.shape[-1] == ynew.shape[-1]
        lags = [lag / 1000 * 512 for lag in args.lags]
        cor_res = torch.empty(yhat.shape[0], yhat.shape[-1])

        if "align-utt-samp" in args.test_mod:
            # Get the smallest number of samples across all lags
            in_utt = df.utt_offset_prev - df.onset <= lags[0]
            in_utt &= df.utt_onset_next - df.onset >= lags[0]
            in_utt2 = df.utt_offset_prev - df.onset <= lags[-1]
            in_utt2 &= df.utt_onset_next - df.onset >= lags[-1]
            in_utt_num = min(in_utt.sum(), in_utt2.sum())

        # in_utt_lags = []
        for lag_idx, lag in enumerate(lags):
            in_utt = df.utt_offset_prev - df.onset <= lag
            in_utt &= df.utt_onset_next - df.onset >= lag
            # print(f"lag: {lag}, in_utt: {in_utt.sum()}")
            # in_utt_lags.append(in_utt.sum())
            yhat_lag = yhat[:, in_utt, lag_idx, np.newaxis]
            ynew_lag = ynew[in_utt, lag_idx, np.newaxis]
            df_lag = df[in_utt].copy()

            if "align-utt-norm" in args.test_mod:
                cor_res_lag = correlation_score_split(ynew_lag, yhat_lag)
                cor_res[:, lag_idx] = (
                    cor_res_lag.squeeze() * yhat_lag.shape[1] / yhat.shape[1]
                )
            elif "align-utt-samp" in args.test_mod:
                if in_utt.sum() >= in_utt_num:
                    samp = 1000
                    cor_res_lags = []
                    for i in np.arange(0, samp):
                        samps = np.random.choice(
                            ynew_lag.shape[0], size=in_utt_num, replace=True
                        )
                        cor_res_lag = correlation_score_split(
                            ynew_lag[samps],
                            yhat_lag[:, samps],
                        )
                        cor_res_lags.append(cor_res_lag.squeeze())
                    cor_res_lags = torch.mean(torch.stack(cor_res_lags), dim=0)
                    cor_res[:, lag_idx] = cor_res_lags

            else:
                cor_res_lag = correlation_score_split_folds(df_lag, ynew_lag, yhat_lag)
                cor_res[:, lag_idx] = cor_res_lag.squeeze()
        # pd.DataFrame([in_utt_lags]).to_csv(f"{args.sid}_comp_lags.csv", index=False)

    return cor_res


def test_cor(args, sid_df, pred_dir, sid, elec_name):

    # Y
    cyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_yhat.npy")
    pyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_yhat.npy")
    cynew = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_ynew.npy")
    pynew = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_ynew.npy")
    if (
        os.path.exists(cyhat)
        and os.path.exists(pyhat)
        and os.path.exists(cynew)
        and os.path.exists(pynew)
    ):
        comp_yhat = np.load(cyhat)
        comp_ynew = np.load(cynew)
        prod_yhat = np.load(pyhat)
        prod_ynew = np.load(pynew)
    else:
        return

    cor_res_comp = correlation_score_split(comp_ynew, comp_yhat)
    save_cor_res(args, f"{sid}_{elec_name}_comp", cor_res_comp)
    cor_res_prod = correlation_score_split(prod_ynew, prod_yhat)
    save_cor_res(args, f"{sid}_{elec_name}_prod", cor_res_prod)
    return


def save_cor_res(args, filename, cor_res):
    for idx, feat in enumerate(FEAT_SPACES):
        if feat == "all":
            try:
                result_df = pd.DataFrame([cor_res.sum(axis=0).cpu().numpy()])
            except:
                result_df = pd.DataFrame([cor_res.sum(axis=0)])
        else:
            try:
                result_df = pd.DataFrame([cor_res[idx, :].cpu().numpy()])
            except:
                result_df = pd.DataFrame([cor_res[idx, :]])
        result_df.to_csv(
            os.path.join(args.output_dir % feat, f"{filename}.csv"),
            index=False,
            header=False,
        )


########################################
########### TEST COR SQUARED ###########
########################################


def test_cor_squared_align(df1, df2, yhat, ynew):
    joint_indices = df1.index.intersection(df2.index)
    if len(df1) != len(df2):
        print("Aligning")
        # Get joint inner indices
        hat_index = df1.index.get_indexer(joint_indices)
        new_index = df2.index.get_indexer(joint_indices)

        # Align df, yhat and ynew using joint indices
        yhat = yhat[:, hat_index, :]
        ynew = ynew[new_index, :]
    df2 = df1.loc[joint_indices]
    return df2, yhat, ynew


def test_cor_squared(args, sid_df, pred_dir, sid, elec_name):

    if os.path.exists(
        os.path.join(args.output_dir % "all", f"{sid}_{elec_name}_comp.csv")
    ) and os.path.exists(
        os.path.join(args.output_dir % "all", f"{sid}_{elec_name}_prod.csv")
    ):
        print(f"Skipping {sid}_{elec_name}")
        return

    # Datum
    cdf = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_df.pkl")
    pdf = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_df.pkl")
    if os.path.exists(cdf) and os.path.exists(pdf):
        comp_df = pd.read_pickle(cdf)
        prod_df = pd.read_pickle(pdf)
    else:
        return
    comp_df.columns = ["word", "onset", "folds"]
    prod_df.columns = ["word", "onset", "folds"]
    comp_df = sid_df.merge(comp_df[["folds"]], left_index=True, right_index=True)
    prod_df = sid_df.merge(prod_df[["folds"]], left_index=True, right_index=True)
    print(f"{args.sid} {elec_name} Comp: {len(comp_df)} Prod: {len(prod_df)}")

    # yhat
    cyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_yhat.npy")
    pyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_yhat.npy")
    if os.path.exists(cyhat) and os.path.exists(pyhat):
        comp_yhat = np.load(cyhat)
        prod_yhat = np.load(pyhat)
    else:
        return

    if "bestlag" in args.test_mod:
        print("Taking the best lag")
        # ynew
        cynew = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_ynew.npy")
        pynew = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_ynew.npy")
        if os.path.exists(cynew) and os.path.exists(pynew):
            comp_ynew = np.load(cynew)
            prod_ynew = np.load(pynew)
        else:
            return

        num_lags = comp_yhat.shape[-1]

        # Get the best lag
        cor_res_comp = filter_cor(args, comp_df, comp_yhat, comp_ynew)
        cor_res_prod = filter_cor(args, prod_df, prod_yhat, prod_ynew)
        cor_max = cor_res_comp.topk(1, dim=1).indices.squeeze()
        comp_yhat = np.stack(
            [comp_yhat[dim, :, idx] for dim, idx in enumerate(cor_max)]
        )
        comp_yhat = np.repeat(comp_yhat[:, :, np.newaxis], num_lags, axis=-1)
        cor_max = cor_res_prod.topk(1, dim=1).indices.squeeze()
        prod_yhat = np.stack(
            [prod_yhat[dim, :, idx] for dim, idx in enumerate(cor_max)]
        )
        prod_yhat = np.repeat(prod_yhat[:, :, np.newaxis], num_lags, axis=-1)

    cor_res_comp = []
    cor_res_prod = []

    all_elecs_sid = pd.read_csv(f"results/tfs/20250508-elecs/{sid}_elecs.csv")
    for electrode in all_elecs_sid.electrode.values:
        # Get electrode info
        elec_name2 = electrode
        # (sid, elec_id2), elec_name2 = electrode
        print(f"\t{elec_name2}")

        # Datum2
        cdf = os.path.join(pred_dir, f"{sid}_{elec_name2}_comp_df.pkl")
        pdf = os.path.join(pred_dir, f"{sid}_{elec_name2}_prod_df.pkl")
        if os.path.exists(cdf) and os.path.exists(pdf):
            comp_df2 = pd.read_pickle(cdf)
            prod_df2 = pd.read_pickle(pdf)
        else:
            print(f"Missing {sid}_{elec_name2} comp or prod")
            continue
        comp_df2.columns = ["word", "onset", "folds"]
        prod_df2.columns = ["word", "onset", "folds"]

        # ynew
        cynew = os.path.join(pred_dir, f"{sid}_{elec_name2}_comp_ynew.npy")
        pynew = os.path.join(pred_dir, f"{sid}_{elec_name2}_prod_ynew.npy")
        if os.path.exists(cynew) and os.path.exists(pynew):
            comp_ynew = np.load(cynew)
            prod_ynew = np.load(pynew)
        else:
            print(f"Missing {sid}_{elec_name2} comp or prod")
            continue

        # Align hat and new
        comp_df2, comp_yhat2, comp_ynew = test_cor_squared_align(
            comp_df, comp_df2, comp_yhat, comp_ynew
        )
        prod_df2, prod_yhat2, prod_ynew = test_cor_squared_align(
            prod_df, prod_df2, prod_yhat, prod_ynew
        )

        cor_res_comp_elec = filter_cor(args, comp_df2, comp_yhat2, comp_ynew)
        cor_res_comp.append(cor_res_comp_elec)
        cor_res_prod_elec = filter_cor(args, prod_df2, prod_yhat2, prod_ynew)
        cor_res_prod.append(cor_res_prod_elec)

    cor_res_comp = torch.stack(cor_res_comp, axis=1)
    save_cor_res_squared(args, f"{sid}_{elec_name}_comp", cor_res_comp)

    cor_res_prod = torch.stack(cor_res_prod, axis=1)
    save_cor_res_squared(args, f"{sid}_{elec_name}_prod", cor_res_prod)

    return


def save_cor_res_squared(args, filename, cor_res):
    for idx, feat in enumerate(FEAT_SPACES):
        if feat == "all":
            try:
                result_df = pd.DataFrame(cor_res.sum(axis=0).cpu().numpy())
            except:
                result_df = pd.DataFrame(cor_res.sum(axis=0))
        else:
            try:
                result_df = pd.DataFrame(cor_res[idx, :, :].cpu().numpy().squeeze())
            except:
                result_df = pd.DataFrame(cor_res[idx, :, :].squeeze())
        result_df.to_csv(
            os.path.join(args.output_dir % feat, f"{filename}.csv"),
            index=False,
            header=False,
        )


########################################
############# TEST COR UTT #############
########################################


def utt_align(args, df):
    lags = [lag / 1000 for lag in args.lags]
    for col in df.columns:  # align to seconds
        if "onset" in col or "offset" in col:
            df[col] = df[col] / 512

    def align_df(df, col):

        df_new = pd.DataFrame()
        for row_idx, row in tqdm(df.groupby(["conv_id", "sen_id"]).first().iterrows()):
            df_new_utt = df.loc[
                (df.onset >= row[col] + lags[0] - 2)
                & (df.onset <= row[col] + lags[-1] + 2),
                ("word", "onset", "offset"),
            ].copy()
            df_new_utt.reset_index(drop=False, inplace=True)
            df_new_utt["lags"] = round((row[col] - df_new_utt.onset) / 0.025)
            df_new_utt["conv_onset"] = row["conv_onset"]
            df_new_utt["conv_offset"] = row["conv_offset"]
            df_new_utt["conv_id"] = row_idx[0]
            df_new_utt["sen_id"] = row_idx[1]
            df_new_utt["conv_name"] = row["conv_name"]
            df_new_utt["speaker"] = row["speaker"]
            df_new_utt["utt_onset"] = row["utt_onset"]
            df_new_utt["utt_offset"] = row["utt_offset"]
            df_new = pd.concat([df_new, df_new_utt], axis=0)
        return df_new

    df_new_onset = align_df(df, "utt_onset")
    df_new_offset = align_df(df, "utt_offset")
    df_new_onset.to_pickle(
        os.path.join(
            args.pred_cache % (args.sid, args.sid), f"{args.sid}_utt_onset_5s.pkl"
        )
    )
    df_new_offset.to_pickle(
        os.path.join(
            args.pred_cache % (args.sid, args.sid), f"{args.sid}_utt_offset_5s.pkl"
        )
    )
    return df_new_onset, df_new_offset


def filter_cor_utt(args, utt_df, ynew, yhat):
    lags = [lag / 25 for lag in args.lags]
    lag_0 = 80  # lag 0 of words
    lag_w = 40  # window to take words (<= 80)
    cor_res = torch.empty(yhat.shape[0], len(lags))
    for lag_idx, lag in enumerate(lags):
        utt_df["word_lag"] = utt_df.lags + lag + lag_0
        utt_df.word_lag = utt_df.word_lag.astype(int)
        lag_utt_df = utt_df.loc[
            utt_df.word_lag.ge(lag_0 - lag_w) & utt_df.word_lag.le(lag_0 + lag_w), :
        ]
        ynew_lag = ynew[lag_utt_df["y_index"], lag_utt_df["word_lag"], np.newaxis]
        yhat_lag = yhat[:, lag_utt_df["y_index"], lag_utt_df["word_lag"], np.newaxis]
        cor_res_lag = correlation_score_split(ynew_lag, yhat_lag)
        cor_res[:, lag_idx] = cor_res_lag.squeeze()

    return cor_res


def test_cor_utt(args, onset_df, offset_df, pred_dir, sid, elec_name):

    if os.path.exists(
        os.path.join(args.output_dir % "all", f"{sid}_{elec_name}_comp.csv")
    ) and os.path.exists(
        os.path.join(args.output_dir % "all", f"{sid}_{elec_name}_prod.csv")
    ):
        print(f"Skipping {sid}_{elec_name}")
        return

    # Datum
    cdf = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_df.pkl")
    pdf = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_df.pkl")
    if os.path.exists(cdf) and os.path.exists(pdf):
        comp_df = pd.read_pickle(cdf)
        prod_df = pd.read_pickle(pdf)
    else:
        return
    comp_df.columns = ["word", "onset"]
    prod_df.columns = ["word", "onset"]

    # Combine the dataframes and filter the onset/offset dataframes
    df = pd.concat([comp_df, prod_df])
    onset_df = onset_df[onset_df["index"].isin(df.index)]
    offset_df = offset_df[offset_df["index"].isin(df.index)]

    print(f"{args.sid} {elec_name} Onset: {len(onset_df)} Offset: {len(offset_df)}")

    # Y
    cyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_yhat.npy")
    pyhat = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_yhat.npy")
    cynew = os.path.join(pred_dir, f"{sid}_{elec_name}_comp_ynew.npy")
    pynew = os.path.join(pred_dir, f"{sid}_{elec_name}_prod_ynew.npy")
    if (
        os.path.exists(cyhat)
        and os.path.exists(pyhat)
        and os.path.exists(cynew)
        and os.path.exists(pynew)
    ):
        comp_yhat = np.load(cyhat)
        comp_ynew = np.load(cynew)
        prod_yhat = np.load(pyhat)
        prod_ynew = np.load(pynew)
    else:
        return

    # Combine the ys using the sorted indices
    ynew = np.concatenate([comp_ynew, prod_ynew], axis=0)
    yhat = np.concatenate([comp_yhat, prod_yhat], axis=1)
    sorted_indices = df.index.argsort()

    ynew = ynew[sorted_indices]
    yhat = yhat[:, sorted_indices, :]
    df = df.sort_index()

    # Add the y indices to onset/offset dataframes
    df = df.reset_index().reset_index()
    df = df.rename(columns={"level_0": "y_index"})
    onset_df = onset_df.merge(
        df[["y_index", "index", "word"]], on=["index", "word"], how="left"
    )
    offset_df = offset_df.merge(
        df[["y_index", "index", "word"]], on=["index", "word"], how="left"
    )

    if "-start" in args.test_mod:
        comp_res = filter_cor_utt(
            args, onset_df.loc[onset_df.speaker != "Speaker1"].copy(), ynew, yhat
        )
        prod_res = filter_cor_utt(
            args, onset_df.loc[onset_df.speaker == "Speaker1"].copy(), ynew, yhat
        )
    if "-end" in args.test_mod:
        comp_res = filter_cor_utt(
            args, offset_df.loc[offset_df.speaker != "Speaker1"].copy(), ynew, yhat
        )
        prod_res = filter_cor_utt(
            args, offset_df.loc[offset_df.speaker == "Speaker1"].copy(), ynew, yhat
        )
    save_cor_res(args, f"{sid}_{elec_name}_comp", comp_res)
    save_cor_res(args, f"{sid}_{elec_name}_prod", prod_res)
    return


@main_timer
def main():

    # Read command line arguments
    args, yml_args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    electrode_info = process_electrodes(args)
    pred_dir = args.pred_cache % (args.sid, args.sid)
    sid_df = pd.read_pickle(f"{args.sid}_datum.pkl")
    sid_df.columns = [
        "word",
        "onset",
        "offset",
        "conv_onset",
        "conv_offset",
        "conv_id",
        "conv_name",
        "sen_id",
        "speaker",
        "utt_onset",
        "utt_offset",
        "utt_offset_prev",
        "utt_onset_next",
        "word_idx",
        "word_idx_r",
    ]

    if "utt-level" in args.test_mod:
        utt_onset_dir = os.path.join(pred_dir, f"{args.sid}_utt_onset_5s.pkl")
        utt_offset_dir = os.path.join(pred_dir, f"{args.sid}_utt_offset_5s.pkl")
        if os.path.exists(utt_onset_dir) and os.path.exists(utt_offset_dir):
            utt_onset_df = pd.read_pickle(utt_onset_dir)
            utt_offset_df = pd.read_pickle(utt_offset_dir)
        else:
            utt_onset_df, utt_offset_df = utt_align(args, sid_df)

    for electrode in electrode_info.items():
        # Get electrode info
        print(electrode)
        (sid, elec_id), elec_name = electrode
        test_cor(args, sid_df, pred_dir, sid, elec_name)

    return


if __name__ == "__main__":
    main()
