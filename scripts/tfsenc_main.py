import csv
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tfsenc_config import setup_environ
from tfsenc_load_signal import load_electrode_data
from tfsenc_parser import parse_arguments
from tfsenc_read_datum import read_datum
from tfsenc_utils import (
    build_XY,
    get_groupkfolds,
    get_kfolds,
    run_regression,
    write_encoding_results,
)
from utils import load_pickle, main_timer, write_config


def get_cpu_count(min_cpus=2):
    if os.getenv("SLURMD_NODENAME"):
        min_cpus = cpu_count()

    return min_cpus


def return_stitch_index(args):
    """[summary]
    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_file = os.path.join(args.PICKLE_DIR, args.stitch_file)
    stitch_index = [0] + load_pickle(stitch_file)
    return stitch_index


def process_subjects(args):
    """Process electrodes for subjects (requires electrode list or sig elec file)

    Args:
        args (namespace): commandline arguments

    Returns:
        electrode_info (dict): Dictionary in the format (sid, elec_id): elec_name
    """
    ds = load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file))
    df = pd.DataFrame(ds)
    df["subject"] = df.subject.astype("int64")

    if args.sig_elec_file:  # sig elec files for 1 or more sid (used for 777)
        sig_elec_file = os.path.join(
            os.path.join(os.getcwd(), "data", args.sig_elec_file)
        )
        sig_elec_list = pd.read_csv(sig_elec_file).rename(
            columns={"electrode": "electrode_name"}
        )
        sid_sig_elec_list = pd.merge(
            df, sig_elec_list, how="inner", on=["subject", "electrode_name"]
        )
        assert len(sig_elec_list) == len(sid_sig_elec_list), "Sig Elecs Missing"
        electrode_info = {
            (values["subject"], values["electrode_id"]): values["electrode_name"]
            for _, values in sid_sig_elec_list.iterrows()
        }

    else:  # electrode list for 1 sid
        assert args.electrodes, "Need electrode list since no sig_elec_list"
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
            for key in args.electrodes
        }

    return electrode_info


def single_electrode_encoding(electrode, args, datum, stitch_index):
    """Doing encoding for one electrode

    Args:
        electrode: tuple in the form ((sid, elec_id), elec_name)
        args (namespace): commandline arguments
        datum: datum of words
        stitch_index: stitch_index

    Returns:
        tuple in the format (sid, electrode name, production len, comprehension len)
    """
    # Get electrode info
    (sid, elec_id), elec_name = electrode

    if elec_name is None:
        print(f"Electrode ID {elec_id} does not exist")
        return (args.sid, None, 0, 0)

    # Load signal Data
    elec_signal, missing_convos = load_electrode_data(
        args, sid, elec_id, stitch_index, False
    )

    # Modify datum based on signal
    if len(missing_convos) > 0:  # signal missing convos
        elec_datum = datum.loc[
            ~datum["conversation_name"].isin(missing_convos)
        ]  # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0:  # datum has no words, meaning no signal
        print(f"{args.sid} {elec_name} No Signal")
        return (args.sid, elec_name, 0, 0)

    # Build design matrices
    X, Y = build_XY(args, elec_datum, elec_signal)

    # Split into production and comprehension
    prod_X = X[elec_datum.speaker == "Speaker1", :]
    comp_X = X[elec_datum.speaker != "Speaker1", :]
    prod_Y = Y[elec_datum.speaker == "Speaker1", :]
    comp_Y = Y[elec_datum.speaker != "Speaker1", :]

    # get folds
    if args.project_id == "podcast":  # podcast
        fold_cat_prod = []
        fold_cat_comp = get_kfolds(comp_X, args.fold_num)
    elif (
        "single-conv" in args.datum_mod or args.conversation_id or args.sid == 798
    ):  # 1 conv
        fold_cat_prod = get_kfolds(prod_X, args.fold_num)
        fold_cat_comp = get_kfolds(comp_X, args.fold_num)
    elif (
        args.project_id == "tfs"
        and elec_datum.conversation_id.nunique() < args.fold_num
    ):  # num of convos less than num of folds (special case for 7170)
        print(f"{args.sid} {elec_name} has less conversations than the number of folds")
        return (args.sid, elec_name, 1, 1)
    else:
        # Get groupkfolds
        fold_cat_prod, fold_cat_comp = get_groupkfolds(elec_datum, X, Y, args.fold_num)
        if (
            len(np.unique(fold_cat_prod)) < args.fold_num
            or len(np.unique(fold_cat_comp)) < args.fold_num
        ):  # need both prod/comp words in all folds
            print(f"{args.sid} {elec_name} failed groupkfold")
            return (args.sid, elec_name, 1, 1)

    elec_name = str(sid) + "_" + elec_name
    print(f"{args.sid} {elec_name} Prod: {len(prod_X)} Comp: {len(comp_X)}")

    # Run regression and save correlation results
    prod_train = prod_X, prod_Y, fold_cat_prod
    comp_train = comp_X, comp_Y, fold_cat_comp

    if args.model_mod and "pc-flip" in args.model_mod:  # flip test
        prod_test, comp_test = comp_train, prod_train
    else:
        prod_test, comp_test = prod_train, comp_train

    if len(prod_train[0]) > 0 and len(prod_test[0]) > 0:
        prod_results = run_regression(args, *prod_train, *prod_test)
        write_encoding_results(args, prod_results, elec_name, "prod")
    if len(comp_train[0]) > 0 and len(comp_test[0]) > 0:
        comp_results = run_regression(args, *comp_train, *comp_test)
        write_encoding_results(args, comp_results, elec_name, "comp")
    return (sid, elec_name, len(prod_X), len(comp_X))


def parallel_encoding(args, electrode_info, datum, stitch_index, parallel=True):
    """Doing encoding for all electrodes in parallel

    Args:
        args (namespace): commandline arguments
        electrode_info: dictionary of electrode id and electrode names
        datum: datum of words
        stitch_index: stitch_index
        parallel: whether to encode for all electrodes in parallel or not

    Returns:
        None
    """

    # if args.emb_type == "gpt2-xl" and args.sid == 676:
    #     parallel = False
    if parallel:
        print("Running all electrodes in parallel")
        summary_file = os.path.join(args.full_output_dir, "summary.csv")  # summary file
        p = Pool(processes=get_cpu_count())  # multiprocessing
        
        # Skipping elecs already done
        if os.path.exists(summary_file): # previous job
            print("Previously ran the same job, checking for elecs done")
            elecs_done = [os.path.basename(file) for file in glob.glob(os.path.join(args.full_output_dir,"*_*.csv"))]
            elecs_done = sorted(elecs_done)
            while len(elecs_done) > 0:
                if "comp" in elecs_done[0]: # has comp
                    if elecs_done[1].replace("prod","comp") == elecs_done[0]: # has prod
                        elec_done = elecs_done.pop(1)
                        elec_done = elec_done.replace("_prod.csv","")
                        if str(args.sid) in elec_done:
                            elec_done = elec_done.replace(f"{args.sid}_","")
                        print(f"Skipping {elec_done}")
                        electrode_info = {key:val for key, val in electrode_info.items() if val != elec_done}
                elecs_done.pop(0)
        
        with open(summary_file, "w") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\r\n")
            writer.writerow(("sid", "electrode", "prod", "comp"))
            for result in p.map(
                partial(
                    single_electrode_encoding,
                    args=args,
                    datum=datum,
                    stitch_index=stitch_index,
                ),
                electrode_info.items(),
            ):
                writer.writerow(result)
    else:
        print("Running all electrodes")
        for electrode in electrode_info.items():
            single_electrode_encoding(electrode, args, datum, stitch_index)

    return None


@main_timer
def main():

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)

    # Processing significant electrodes or individual subjects
    electrode_info = process_subjects(args)
    parallel_encoding(args, electrode_info, datum, stitch_index)

    return


if __name__ == "__main__":
    main()
