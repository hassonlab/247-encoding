import os

import numpy as np
import pandas as pd
from utils import load_pickle


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding
    """
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())
    df = df[~df['is_nan']]

    return df


def return_stitch_index(args):
    """[summary]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_file = os.path.join(args.PICKLE_DIR, args.stitch_file)
    stitch_index = load_pickle(stitch_file)
    return stitch_index


def adjust_onset_offset(args, df):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)
    assert len(stitch_index) == df.conversation_id.nunique()

    stitch_index = [0] + stitch_index[:-1]

    df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        shift = stitch_index[idx]
        df.loc[df.conversation_id == conv,
               'onset'] = df.loc[df.conversation_id == conv,
                                 'adjusted_onset'] - shift
        df.loc[df.conversation_id == conv,
               'offset'] = df.loc[df.conversation_id == conv,
                                  'adjusted_offset'] - shift

    return df


def add_signal_length(args, df):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)

    signal_lengths = np.diff(stitch_index).tolist()
    signal_lengths.insert(0, stitch_index[0])

    df['conv_signal_length'] = np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        df.loc[df.conversation_id == conv,
               'conv_signal_length'] = signal_lengths[idx]

    return df


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """

    file_name = os.path.join(args.PICKLE_DIR, args.load_emb_file)
    datum = load_pickle(file_name)

    df = pd.DataFrame.from_dict(datum)
    df = add_signal_length(args, df)

    if args.project_id == 'tfs' and not all(
        [item in df.columns
         for item in ['adjusted_onset', 'adjusted_offset']]):
        df = adjust_onset_offset(args, df)
    df = drop_nan_embeddings(df)

    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]

    # use columns where token is root
    if 'gpt2' in [args.align_with, args.emb_type]:
        df = df[df['gpt2_token_is_root']]
    elif 'bert' in [args.align_with, args.emb_type]:
        df = df[df['bert_token_is_root']]
    else:
        pass

    df = df[~df['glove50_embeddings'].isna()]

    # if encoding is on glove embeddings copy them into 'embeddings' column
    if args.emb_type == 'glove50':
        df['embeddings'] = df['glove50_embeddings']

    return df
