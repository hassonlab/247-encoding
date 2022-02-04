import os
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle

NONWORDS = {'hm', 'huh', 'mhm', 'mm', 'oh', 'uh', 'uhuh', 'um'}


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


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


def make_input_from_tokens(token_list):
    """[summary]

    Args:
        args ([type]): [description]
        token_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    windows = [
        tuple(token_list[x:x + 2]) for x in range(len(token_list) - 2 + 1)
    ]

    return windows


def add_convo_onset_offset(args, df):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_index = return_stitch_index(args)
    assert len(stitch_index) == df.conversation_id.nunique()

    stitch_index = [0] + stitch_index
    windows = make_input_from_tokens(stitch_index)

    df['convo_onset'], df['convo_offset'] = np.nan, np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        edges = windows[idx]

        df.loc[df.conversation_id == conv, 'convo_onset'] = edges[0]
        df.loc[df.conversation_id == conv, 'convo_offset'] = edges[1]

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


def normalize_embeddings(args, df):
    """Normalize the embeddings
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

    Args:
        args ([type]): [description]
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    k = np.array(df.embeddings.tolist())

    try:
        k = normalize(k, norm=args.normalize, axis=1)
    except ValueError:
        df['embeddings'] = k.tolist()

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
    start_n = len(df)

    if args.project_id == 'tfs' and not all(
        [item in df.columns
         for item in ['adjusted_onset', 'adjusted_offset']]):
        df = adjust_onset_offset(args, df)
    # TODO this was needed for podcast but destroys tfs
    # else:
    #     df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    #     df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    df = add_convo_onset_offset(args, df)
    if args.emb_type == 'glove50':
        df = df.dropna(subset=['embeddings'])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)
    # df = df[~df['glove50_embeddings'].isna()]

    # Apply filters
    common = []
    for model in args.align_with:
        if 'gpt2' in model:
            common.append(df.in_gpt2.values)
        elif 'blenderbot-small' in model:
            common.append(df.in_blenderbot_small_90M.values)
        elif 'glove' in model:
            common.append(df.in_glove.values)
        elif 'blenderbot' in model:
            common.append(df.in_blenderbot_3B.values)

    if 'gpt2' in args.emb_type.lower():
        common.append(df.in_gpt2.values)
    elif 'blenderbot-small' in args.emb_type.lower():
        common.append(df.in_blenderbot_small_90M.values)
    elif 'blenderbot' in args.emb_type.lower():
        common.append(df.in_blenderbot_3B.values)

    if args.exclude_nonwords:
        nonword_mask = df.word.str.lower().apply(lambda x: x in NONWORDS)
        common.append(~nonword_mask.values)

    if args.min_word_freq > 0:
        freq_mask = df.word_freq_overall >= args.min_word_freq
        common.append(freq_mask)

    # # NOTE  something happening here, in_gpt2 messed up apparently
    # if len(common):
    #     common = np.vstack(common)
    #     df = df[common.all(axis=0)].copy()

    end_n = len(df)
    print(f'Went from {start_n} words to {end_n} words')

    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]
        df.convo_offset = df['convo_offset'] - df['convo_onset']
        df.convo_onset = 0

    # if encoding is on glove embeddings copy them into 'embeddings' column
    if args.emb_type == 'glove50':
        try:
            df['embeddings'] = df['glove50_embeddings']
        except KeyError:
            pass

    if not args.normalize:
        df = normalize_embeddings(args, df)

    return df
