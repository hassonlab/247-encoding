import csv
import glob
import os
import torch

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tfsenc_config import parse_arguments, setup_environ, write_config
from tfsenc_read_datum import read_datum
from tfsenc_load_signal import load_electrode_data
from tfsenc_encoding import (
    encoding_setup,
    run_encoding,
    write_encoding_results,
)
from utils import load_pickle, main_timer


def return_stitch_index(args):
    """Return convo onset and offset boundaries
    Args:
        args (namespace): arguments

    Returns:
        stitch_index (list): list of convo boundaries
    """
    stitch_index = [0] + load_pickle(args.stitch_file_path)
    return stitch_index


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
        assert len(args.elecs > 0), "Need electrode list since no sig_elec_list"
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


def skip_elecs_done(summary_file, electrode_info, args):
    """Skip electrodes with encoding results already

    Args:
        summary_file (string): path to summary file of previous jobs
        electrode_info (dict): dictionary of electrodes

    Returns:
        electrode_info (dict): dictionary of electrodes without electrodes with encoding results
    """
    elecs_num = len(electrode_info)
    #summary = pd.read_csv(summary_file, header=None)

    #elecs_done = summary.iloc[:, 1].tolist()
    #elecs_done = [elec for elec in elecs_done if elec not in ['SG1', 'SG2']]
    if getattr(args, "permute", False):
        elecs_done = [os.path.split(fname)[-1].split("_prod")[0]
                    for fname in glob.glob(os.path.join(os.path.split(summary_file)[0], "*_prod_perm_split.npz"))]
    else:
        elecs_done = [os.path.split(fname)[-1].split("_prod")[0]
                    for fname in glob.glob(os.path.join(os.path.split(summary_file)[0], "*_prod_split.npz"))]
    for elec in elecs_done:  # skipping electrodes
        print(f"Skipping elec {elec}")
        electrode_info = {
            key: val for key, val in electrode_info.items() if f"{key[0]}_{val}" != elec
        }
        elecs_num -= 1
    print(f"**************Electrode info: {electrode_info}")
    #assert elecs_num == len(electrode_info), "Wrong number of elecs skipped"
    return electrode_info

#@profile
def single_electrode_encoding(electrode, args, datum, stitch_index):
    """Doing encoding for one electrode

    Args:
        electrode (tuple): ((sid, elec_id), elec_name)
        args (namespace): commandline arguments
        datum (df): datum of words
        stitch_index (list): stitch_index

    Returns:
        tuple in the format (sid, electrode name, production len, comprehension len)
    """
    # Get electrode info
    (sid, elec_id), elec_name = electrode

    if elec_name is None:
        print(f"Electrode ID {elec_id} does not exist")
        return (args.sid, None, 0, 0)

    # Load signal Data
    elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index)
    if len(missing_convos) > 0:  # modify datum based on missing signal
        elec_datum = datum.loc[
            ~datum["conversation_name"].isin(missing_convos)
        ]  # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0:  # datum has no words, meaning no signal
        print(f"{args.sid} {elec_name} No Signal")
        return (args.sid, elec_name, 0, 0)

    # Set up encoding (prod/comp x, y, and folds)
    comp_data, prod_data, extra_train_comp_data, extra_train_prod_data, extra_test_comp_data, extra_test_prod_data = encoding_setup(
        args, elec_name, elec_datum, elec_signal
    )
    elec_name = str(sid) + "_" + elec_name
    print(f"{args.sid} {elec_name} Comp: {len(comp_data[0])} Prod: {len(prod_data[0])}")

    # Run encoding and save results
    if args.comp and len(comp_data[0]) > 0:  # Comprehension
        if len(np.unique(comp_data[2])) < args.cv_fold_num:
            print(f"{args.sid} {elec_name} failed comp groupkfold")
        if np.nansum(comp_data[1]) == 0:
            print(f"{args.sid} {elec_name} comp all NaNs")
        else:
            if "permute" in args and args.permute:
                result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, _, _ = run_encoding(
                    args, *comp_data,
                    extra_train_data=extra_train_comp_data,
                    extra_test_data=extra_test_comp_data,
                    permute=True
                )
                write_encoding_results(
                    args, result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, f"{elec_name}_comp_perm.csv", folds=prod_data[-1]
                )
            else:
                result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, all_fold_yhat_split, params = run_encoding(
                    args, *comp_data,
                    extra_train_data=extra_train_comp_data ,
                    extra_test_data=extra_test_comp_data,
                )
                write_encoding_results(
                    args, result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, f"{elec_name}_comp.csv", folds=comp_data[-1], all_fold_yhat_split=all_fold_yhat_split,
                    params=params
                )
    if args.prod and len(prod_data[0]) > 0:  # Production
        if len(np.unique(prod_data[2])) < args.cv_fold_num:
            print(f"{args.sid} {elec_name} failed prod groupkfold")
        if np.nansum(prod_data[1]) == 0:
            print(f"{args.sid} {elec_name} prod all NaNs")
        else:
            if "permute" in args and args.permute:
                result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, _, _ = run_encoding(
                    args, *prod_data,
                    extra_train_data=extra_train_prod_data,
                    extra_test_data=extra_test_prod_data,
                    permute=True
                )
                write_encoding_results(
                    args, result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, f"{elec_name}_prod_perm.csv", folds=prod_data[-1], params=params
                )
            else:
                result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, all_fold_yhat_split, params = run_encoding(
                    args, *prod_data,
                    extra_train_data=extra_train_prod_data,
                    extra_test_data=extra_test_prod_data
                )
                write_encoding_results(
                    args, result, result_split, Y_hat, Y_new, Y_hat_extra, Y_new_extra, f"{elec_name}_prod.csv", folds=prod_data[-1], all_fold_yhat_split=all_fold_yhat_split,
                    params=params
                )

    return (sid, elec_name, len(prod_data[0]), len(comp_data[0]))

#@profile
def electrodes_encoding(args, electrode_info, datum, stitch_index):
    """Doing encoding for all electrodes

    Args:
        args (namespace): commandline arguments
        electrode_info (dict): dictionary of electrodes
        datum (df): datum of words
        stitch_index (list): stitch_index
    """

    summary_file = os.path.join(args.output_dir, "summary.csv")  # summary file
    if False: #os.path.exists(summary_file):  # previous job
        print("Previously ran the same job, checking for elecs done")
        electrode_info = skip_elecs_done(summary_file, electrode_info, args)

    for electrode in list(electrode_info.items())[::-1]:
        try:
            result = single_electrode_encoding(electrode, args, datum, stitch_index)
            with open(summary_file, "a") as f:
                writer = csv.writer(f, delimiter=",", lineterminator="\r\n")
                writer.writerow(result)
        except Exception as e:
            print(f"FAILED on {electrode}: {e}")

    return None


#@profile
@main_timer
def main():

    test = torch.ones(5000).cuda()
    print(test)
    
    # Read command line arguments
    args, yml_args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(args, yml_args)

    # Locate and read datum
    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)

    # Processing significant electrodes or individual subjects
    electrode_info = process_electrodes(args)
    electrodes_encoding(args, electrode_info, datum, stitch_index)

    return


if __name__ == "__main__":
    main()
