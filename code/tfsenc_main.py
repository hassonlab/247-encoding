import csv
import glob
import os
from functools import partial
from multiprocessing import Pool
from urllib.parse import _NetlocResultMixinBytes

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tfsenc_parser import parse_arguments
from tfsenc_phase_shuffle import phase_randomize_1d
from tfsenc_read_datum import read_datum
from tfsenc_load_signal import load_electrode_data
from tfsenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header, setup_environ)
from utils import load_pickle, main_timer, write_config


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


# def process_datum(args, df):
#     df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

#     # drop empty embeddings
#     df = df[~df['is_nan']]

#     # use columns where token is root
#     if 'gpt2-xl' in [args.align_with, args.emb_type]:
#         df = df[df['gpt2-xl_token_is_root']]
#     elif 'bert' in [args.align_with, args.emb_type]:
#         df = df[df['bert_token_is_root']]
#     else:
#         pass

#     df = df[~df['glove50_embeddings'].isna()]

#     if args.emb_type == 'glove50':
#         df['embeddings'] = df['glove50_embeddings']

#     return df


# def load_processed_datum(args):
#     conversations = sorted(
#         glob.glob(
#             os.path.join(os.getcwd(), 'data', str(args.sid), 'conv_embeddings',
#                          '*')))
#     all_datums = []
#     for conversation in conversations:
#         datum = load_pickle(conversation)
#         df = pd.DataFrame.from_dict(datum)
#         df = process_datum(args, df)
#         all_datums.append(df)

#     concatenated_datum = pd.concat(all_datums, ignore_index=True)

#     return concatenated_datum


def process_subjects(args):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    # trimmed_signal = trimmed_signal_dict['trimmed_signal']

    # if args.electrodes:
    #     indices = [electrode_ids.index(i) for i in args.electrodes]

    #     trimmed_signal = trimmed_signal[:, indices]
    #     electrode_names = [electrode_names[i] for i in indices]

    ds = load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file))
    df = pd.DataFrame(ds)

    if args.electrodes:
        electrode_info = {
            key: next(
                iter(df.loc[(df.subject == str(args.sid)) &
                            (df.electrode_id == key), 'electrode_name']), None)
            for key in args.electrodes
        }

    # # Loop over each electrode
    # for elec_id, elec_name in electrode_info.items():

    #     if elec_name is None:
    #         print(f'Electrode ID {elec_id} does not exist')
    #         continue

    #     elec_signal = load_electrode_data(args, elec_id)
    #     # datum = load_processed_datum(args)

    #     encoding_regression(args, datum, elec_signal, elec_name)

    # # write_electrodes(args, electrode_names)

    return electrode_info


def process_sig_electrodes(args, datum):
    """Run encoding on select significant elctrodes specified by a file
    """
    # Read in the significant electrodes
    sig_elec_file = os.path.join(
        os.path.join(os.getcwd(), 'data', args.sig_elec_file))
    sig_elec_list = pd.read_csv(sig_elec_file)

    # Loop over each electrode
    for subject, elec_name in sig_elec_list.itertuples(index=False):

        assert isinstance(subject, int)
        CONV_DIR = '/projects/HASSON/247/data/conversations'
        if args.project_id == 'podcast':
            CONV_DIR = '/projects/HASSON/247/data/podcast'
        BRAIN_DIR_STR = 'preprocessed_all'

        fname = os.path.join(CONV_DIR, 'NY' + str(subject) + '*')
        subject_id = glob.glob(fname)
        assert len(subject_id), f'No data found in {fname}'
        subject_id = os.path.basename(subject_id[0])

        # Read subject's header
        labels = load_header(CONV_DIR, subject_id)
        assert labels is not None, 'Missing header'
        electrode_num = labels.index(elec_name) + 1

        # Read electrode data
        brain_dir = os.path.join(CONV_DIR, subject_id, BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                subject_id, '_electrode_preprocess_file_',
                str(electrode_num), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
            elec_signal = elec_signal.reshape(-1, 1)
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue

        # Perform encoding/regression
        encoding_regression(args, datum, elec_signal,
                            str(subject) + '_' + elec_name)

    return


def dumdum1(iter_idx, args, datum, signal, name):
    seed = iter_idx + (os.getenv("SLURM_ARRAY_TASk_ID", 0) * 10000)
    np.random.seed(seed)
    new_signal = phase_randomize_1d(signal)
    (prod_corr, comp_corr) = encoding_regression_pr(args, datum, new_signal,
                                                    name)

    return (prod_corr, comp_corr)


def write_output(args, output_mat, name, output_str):

    output_dir = create_output_directory(args)

    if all(output_mat):
        trial_str = append_jobid_to_string(args, output_str)
        filename = os.path.join(output_dir, name + trial_str + '.csv')
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(output_mat)


def this_is_where_you_perform_regression(electrode, args, datum, stitch_index):

    elec_id, elec_name = electrode # get electrode info

    if elec_name is None:
        print(f'Electrode ID {elec_id} does not exist')
        return None

    elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index, False)

    if len(missing_convos) > 0: # signal missing convos
        elec_datum = datum.loc[~datum['conversation_name'].isin(missing_convos)] # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0: # no signal
        print(f'{args.sid} {elec_name} No Signal')
        return None
    elif elec_datum.conversation_id.nunique() < 5: # less than 5 convos
        print(f'{args.sid} {elec_name} has less than 5 conversations')
        return None

    # Perform encoding/regression
    if args.phase_shuffle:
        if args.project_id == 'podcast':
            with Pool() as pool:
                corr = pool.map(
                    partial(dumdum1,
                            args=args,
                            datum=elec_datum,
                            signal=elec_signal,
                            name=elec_name), range(args.npermutations))
        else:
            corr = []
            for i in range(args.npermutations):
                corr.append(dumdum1(i, args, elec_datum, elec_signal,
                                    elec_name))

        prod_corr, comp_corr = map(list, zip(*corr))
        write_output(args, prod_corr, elec_name, 'prod')
        write_output(args, comp_corr, elec_name, 'comp')
    else:
        encoding_regression(args, elec_datum, elec_signal, elec_name)

    return None


def parallel_regression(args, electrode_info, datum, stitch_index):
    parallel = True
    if args.emb_type == 'gpt2-xl' and args.sid == 676:
        parallel = False
    if parallel:
        print('Running all electrodes in parallel')
        with Pool(4) as p:
            p.map(
                partial(this_is_where_you_perform_regression,
                    args = args,
                    datum = datum,
                    stitch_index = stitch_index,
                ), electrode_info.items())
    else:
        print('Running all electrodes')
        for electrode in electrode_info.items():
            this_is_where_you_perform_regression(electrode, args, datum, stitch_index)

    return None

def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
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
    if args.sig_elec_file:
        process_sig_electrodes(args, datum)
    else:
        electrode_info = process_subjects(args)
        parallel_regression(args, electrode_info, datum, stitch_index)

    return


if __name__ == "__main__":
    main()
