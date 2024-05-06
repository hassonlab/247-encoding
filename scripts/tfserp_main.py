import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tfsenc_load_signal import load_electrode_data
from tfsenc_main import process_subjects, return_stitch_index
from tfsenc_parser import parse_arguments
from tfsenc_read_datum import read_datum
from tfsenc_config import setup_environ
from tfsenc_utils import append_jobid_to_string
from utils import main_timer, write_config


def erp(args, datum, elec_signal, name):
    datum = datum[datum.adjusted_onset.notna()]

    datum_comp = datum[datum.speaker != "Speaker1"]  # comprehension data
    datum_prod = datum[datum.speaker == "Speaker1"]  # production data
    print(
        f"{args.sid} {name} Prod: {len(datum_comp.index)} Comp: {len(datum_prod.index)}"
    )

    erp_comp = calc_average(args.lags, datum_comp, elec_signal)  # calculate average erp
    erp_prod = calc_average(args.lags, datum_prod, elec_signal)  # calculate average erp

    print(f"writing output for electrode {name}")
    write_erp_results(args, erp_comp, name, "comp")
    write_erp_results(args, erp_prod, name, "prod")

    return


def calc_average(lags, datum, brain_signal):
    """[summary]
    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    onsets = datum.adjusted_onset.values
    erp = np.zeros((len(onsets), len(lags)))

    for lag_idx, lag in enumerate(lags):  # loop through each lag
        lag_amount = int(lag / 1000 * 512)
        index_onsets = (
            np.round_(onsets, 0, onsets) + lag_amount
        )  # take correct idx for all words
        index_onsets = index_onsets.astype(int)  # uncomment this if not running jit
        erp[:, lag_idx] = brain_signal[index_onsets].reshape(
            -1
        )  # take the signal for that lag

    erp = [np.mean(erp, axis=(0), dtype=np.float64).tolist()]  # average by words

    return erp


def write_erp_results(args, erp_results, elec_name, mode):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        erp_results: erp results
        elec_name: electrode name as a substring of filename
        mode: 'prod' or 'comp'

    Returns:
        None
    """
    trial_str = append_jobid_to_string(args, mode)
    filename = os.path.join(args.full_output_dir, elec_name + trial_str + ".csv")

    cor_datum = erp_results[0]
    with open(filename, "w") as csvfile:
        print("writing file")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(cor_datum)

    return None


def load_and_erp(electrode, args, datum, stitch_index):
    """Doing ERP for one electrode

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
    elec_name = str(sid) + "_" + elec_name

    # load electrode signal (with z_score)
    elec_signal, missing_convos = load_electrode_data(
        args, sid, elec_id, stitch_index, True
    )
    elec_signal = elec_signal.reshape(-1, 1)

    # trim datum based on signal
    if len(missing_convos) > 0:  # signal missing convos
        elec_datum = datum.loc[
            ~datum["conversation_name"].isin(missing_convos)
        ]  # filter missing convos
    else:
        elec_datum = datum

    # special cases for missing signal
    if len(elec_datum) == 0:  # no signal
        print(f"{args.sid} {elec_name} No Signal")
        return (args.sid, elec_name, 0, 0)

    # do and save erp
    erp(args, elec_datum, elec_signal, elec_name)

    return


def load_and_erp_parallel(args, electrode_info, datum, stitch_index, parallel=True):
    """Doing ERP for all electrodes in parallel

    Args:
        args (namespace): commandline arguments
        electrode_info: dictionary of electrode id and electrode names
        datum: datum of words
        stitch_index: stitch_index
        parallel: whether to encode for all electrodes in parallel or not

    Returns:
        None
    """
    if parallel:
        print("Running all electrodes in parallel")
        summary_file = os.path.join(args.full_output_dir, "summary.csv")  # summary file
        p = Pool(4)  # multiprocessing

        with open(summary_file, "w") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\r\n")
            writer.writerow(("sid", "electrode", "prod", "comp"))
            for result in p.map(
                partial(
                    load_and_erp,
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
            load_and_erp(electrode, args, datum, stitch_index)


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
    datum = datum.drop("embeddings", axis=1)  # trim datum to smaller size

    # Process and do ERP
    assert args.sig_elec_file == None, "Do not input significant electrode list"
    electrode_info = process_subjects(args)
    load_and_erp_parallel(args, electrode_info, datum, stitch_index)

    return


if __name__ == "__main__":
    main()
