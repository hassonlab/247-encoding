import os
import string
from multiprocessing.spawn import is_forking

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle

# import gensim.downloader as api
# import re

NONWORDS = {"hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"}


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    df["is_nan"] = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~df["is_nan"]]

    return df


def adjust_onset_offset(args, df, stitch_index):
    """[summary]

    Args:
        args ([type]): [description]
        df ([type]): [description]
        stitch_index ([list]): stitch index

    Returns:
        [type]: [description]
    """
    print("adjusting datum onset and offset")
    stitch_index = stitch_index[:-1]

    df["adjusted_onset"], df["onset"] = df["onset"], np.nan
    df["adjusted_offset"], df["offset"] = df["offset"], np.nan

    for _, conv in enumerate(df.conversation_id.unique()):
        shift = stitch_index[conv - 1]
        df.loc[df.conversation_id == conv, "onset"] = (
            df.loc[df.conversation_id == conv, "adjusted_onset"] - shift
        )
        df.loc[df.conversation_id == conv, "offset"] = (
            df.loc[df.conversation_id == conv, "adjusted_offset"] - shift
        )
    return df


def make_input_from_tokens(token_list):
    """[summary]

    Args:
        args ([type]): [description]
        token_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    windows = [tuple(token_list[x : x + 2]) for x in range(len(token_list) - 2 + 1)]

    return windows


def add_convo_onset_offset(args, df, stitch_index):
    """Add conversation onset and offset to datum

    Args:
        args (namespace): commandline arguments
        df (DataFrame): datum being processed
        stitch_index ([list]): stitch_index

    Returns:
        Dataframe: df with conversation onset and offset
    """
    windows = make_input_from_tokens(stitch_index)

    df["convo_onset"], df["convo_offset"] = np.nan, np.nan

    for _, conv in enumerate(df.conversation_id.unique()):
        edges = windows[conv - 1]

        df.loc[df.conversation_id == conv, "convo_onset"] = edges[0]
        df.loc[df.conversation_id == conv, "convo_offset"] = edges[1]

    return df


def add_signal_length(df, stitch):
    """Add conversation signal length to datum

    Args:
        df (DataFrame): datum being processed
        stitch (List): stitch index

    Returns:
        DataFrame: df with conversation signal length
    """
    signal_lengths = np.diff(stitch).tolist()

    df["conv_signal_length"] = np.nan

    for idx, conv in enumerate(df.conversation_id.unique()):
        df.loc[df.conversation_id == conv, "conv_signal_length"] = signal_lengths[idx]

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
        df["embeddings"] = k.tolist()

    return df


def load_datum(file_name):
    """Read raw datum

    Args:
        filename: raw datum full file path

    Returns:
        DataFrame: datum
    """
    datum = load_pickle(file_name)
    df = pd.DataFrame.from_dict(datum)
    return df


def process_datum(args, df, stitch):
    """Process the datum based on input arguments

    Args:
        args (namespace): commandline arguments
        df : raw datum as a DataFrame
        stitch: stitch index

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """

    df = add_signal_length(df, stitch)

    df = df.loc[~df["conversation_id"].isin(args.bad_convos)]  # filter bad convos
    assert len(stitch) - len(args.bad_convos) == df.conversation_id.nunique() + 1

    if args.project_id == "tfs" and not all(
        [item in df.columns for item in ["adjusted_onset", "adjusted_offset"]]
    ):
        df = adjust_onset_offset(
            args, df, stitch
        )  # not sure if needed (maybe as a failsafe?)
    # TODO this was needed for podcast but destroys tfs
    # else:
    #     df['adjusted_onset'], df['onset'] = df['onset'], np.nan
    #     df['adjusted_offset'], df['offset'] = df['offset'], np.nan

    df = df[df.adjusted_onset.notna()]
    df = add_convo_onset_offset(args, df, stitch)

    if args.emb_type == "glove":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)

    # df = df[~df['glove50_embeddings'].isna()]
    # if encoding is on glove embeddings copy them into 'embeddings' column
    # if args.emb_type == "glove":
    #     try:
    #         df["embeddings"] = df["glove50_embeddings"]
    #     except KeyError:
    #         pass

    if not args.normalize:
        df = normalize_embeddings(args, df)

    return df


def filter_datum(args, df):
    """Filter the datum based on embedding types, non_words, min_word_freq arguments

    Args:
        args (namespace): commandline arguments
        df: processed datum

    Returns:
        DataFrame: filtered datum
    """
    common = np.repeat(True, len(df))  # create mask for filtering
    # common = df[f"in_{args.emb_type}"]

    # filter based on align with arguments
    for model in args.align_with:
        if model == args.emb_type:  # FIXME: delete this later
            continue
        if model == "glove":  # when aligning with glove
            common = (
                common & df[f"{args.emb_type}_token_is_root"]
            )  # also ensure word=token
        print(f"Aligning with {model}")
        common = common & df[f"in_{model}"]

    if args.exclude_nonwords:  # filter based on exclude_nonwords argument
        nonword_mask = df.word.str.lower().isin(args["non_words"])
        common &= ~nonword_mask

    if args.min_word_freq > 0:  # filter based on min_word_freq argument
        freq_mask = df.word_freq_overall >= args.min_word_freq
        common &= freq_mask

    df = df[common]

    return df


def mod_datum_by_preds(args, datum, emb_type):
    """Filter the datum based on the predictions of a potentially different model

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        emb_type: embedding type needed to filter the datum

    Returns:
        DataFrame: further filtered datum
    """
    if emb_type in args.load_emb_file:  # current datum has the correct emb_type
        pass
    else:  # current datum does not have the correct emb_type, need to load a second datum

        # load second datum
        if emb_type == "gpt2-xl":
            second_datum_name = (
                str(args.sid) + "_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl"
            )
        elif emb_type == "blenderbot-small":
            second_datum_name = (
                str(args.sid) + "_full_blenderbot-small_layer_16_embeddings.pkl"
            )
        second_datum_path = os.path.join(
            args.PICKLE_DIR, second_datum_name
        )  # second datum full path
        second_datum = load_datum(second_datum_path)[
            ["adjusted_onset", "top1_pred"]
        ]  # load second datum

        # merge second datum prediction columns to datum
        datum = datum.drop(
            ["top1_pred"], axis=1, errors="ignore"
        )  # delete the current top predictions if any
        datum = datum[datum.adjusted_onset.notna()]
        second_datum = second_datum[second_datum.adjusted_onset.notna()]
        datum = datum.merge(second_datum, how="inner", on="adjusted_onset")

    # modify datum based on correct or incorrect predictions
    if (
        "incorrect" in args.datum_mod
    ):  # select words predicted by gpt2 incorrectly (top 1 pred)
        datum = datum[
            datum.word.str.lower() != datum.top1_pred.str.lower().str.strip()
        ]  # incorrect
        print(
            f"Selected {len(datum.index)} incorrect words based on {emb_type} predictions"
        )
    elif (
        "correct" in args.datum_mod
    ):  # select words predicted by gpt2 correctly (top 1 pred)
        datum = datum[
            datum.word.str.lower() == datum.top1_pred.str.lower().str.strip()
        ]  # correct
        print(
            f"Selected {len(datum.index)} correct words based on {emb_type} predictions"
        )

    # elif args.datum_mod == emb_type + "-pred": # for incorrectly predicted words, replace with top 1 pred (only used for podcast glove)
    #     glove = api.load('glove-wiki-gigaword-50')
    #     datum['embeddings'] = datum.top1_pred.str.strip().apply(lambda x: get_vector(x.lower(), glove))
    #     datum = datum[datum.embeddings.notna()]
    #     print(f'Changed words into {emb_type} top predictions')
    else:  # exception
        raise Exception("Invalid Datum Modification")

    return datum


def mod_datum_arg_parse(args, mode):
    partial = args.datum_mod[args.datum_mod.find(mode) + len(mode) :]

    if partial.find("-") >= 0:
        partial = partial[: partial.find("-")]
    else:
        pass
    if len(partial) == 0:
        partial = "1"

    step = -1
    if "n" in partial:
        step = 1
        if partial == "n":
            partial = "1"
        else:
            partial = partial[1:]
    assert partial.isdigit()
    shift_num = int(partial)
    print(f"{mode} {shift_num} * {step * -1} steps ")

    return (shift_num, step)


def shift_emb(args, datum, mode="shift-emb"):
    """Shift the embeddings based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        mode: concat-emb

    Returns:
        DataFrame: datum with shifted embeddings
    """
    shift_num, step = mod_datum_arg_parse(args, mode)

    before_shift_num = len(datum.index)
    for i in np.arange(shift_num):
        datum["embeddings"] = datum.embeddings.shift(step)
        if (
            "blenderbot-small" in args.emb_type.lower()
            or "bert" in args.emb_type.lower()
        ):
            datum = datum[
                (
                    datum.production.shift(step) == datum.production
                    and datum.conversation_id.shift(step) == datum.conversation_id
                )
            ]
        else:
            datum = datum[datum.conversation_id.shift(step) == datum.conversation_id]
    print(f"Shifting resulted in {before_shift_num - len(datum.index)} less words")
    return datum


def concat_emb(args, datum, mode="concat-emb"):
    """Concatenate the embeddings based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        mode: concat-emb

    Returns:
        DataFrame: datum with shifted embeddings
    """
    shift_num, step = mod_datum_arg_parse(args, mode)

    before_shift_num = len(datum.index)
    datum["embeddings_shifted"] = datum.embeddings
    for i in np.arange(shift_num):
        datum["embeddings_shifted"] = datum.embeddings_shifted.shift(step)
        if (
            "blenderbot-small" in args.emb_type.lower()
            or "bert" in args.emb_type.lower()
        ):
            datum = datum[
                (
                    datum.production.shift(step) == datum.production
                    and datum.conversation_id.shift(step) == datum.conversation_id
                )
            ]
        else:
            datum = datum[datum.conversation_id.shift(step) == datum.conversation_id]

        def concat(x):
            return np.concatenate((x["embeddings"], x["embeddings_shifted"]))

        datum["embeddings"] = datum.apply(concat, axis=1)

    print(f"Concatenating resulted in {before_shift_num - len(datum.index)} less words")

    return datum


def trim_datum(args, datum):
    """Trim the datum based on the largest lag size

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: datum with trimmed words
    """
    half_window = round((args.window_size / 1000) * 512 / 2)
    lag = int(args.lags[-1] / 1000 * 512)  # trim edges based on lag
    original_len = len(datum.index)
    datum = datum.loc[
        ((datum["adjusted_onset"] - lag) >= (datum["convo_onset"] + half_window + 1))
        & ((datum["adjusted_onset"] + lag) <= (datum["convo_offset"] - half_window - 1))
    ]
    new_datum_len = len(datum.index)
    print(
        f"Trimming resulted in {new_datum_len} ({round(new_datum_len/original_len*100,5)}%) words"
    )
    return datum


def mod_datum(args, datum):
    """Filter the datum based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: further filtered datum
    """
    if args.conversation_id:  # picking single conversation
        datum = datum[datum.conversation_id == args.conversation_id]
        datum.convo_offset = datum["convo_offset"] - datum["convo_onset"]
        datum.convo_onset = 0

    if "notrim" in args.datum_mod:  # no need for edge trimming
        pass
    else:
        datum = trim_datum(args, datum)  # trim edges

    if args.conversation_id:  # picking single conversation
        datum = datum[datum.conversation_id == args.conversation_id]
        datum.convo_offset = datum["convo_offset"] - datum["convo_onset"]
        datum.convo_onset = 0
        print(f"Running conversation {args.conversation_id} with {len(datum)} words")

    if "shift-emb" in args.datum_mod:  # shift embeddings to include word
        datum = shift_emb(args, datum, "shift-emb")
    elif "concat-emb" in args.datum_mod:
        datum = concat_emb(args, datum, "concat-emb")
    else:
        pass

    if "-all" in args.datum_mod:
        pass

    else:  # modify datum based on predictions
        pred_type = args.emb_type
        if "gpt2-xl" in args.datum_mod:
            pred_type = "gpt2-xl"
        elif "blenerbot-small" in args.datum_mod:
            pred_type = "blenderbot-small"
        assert "glove" not in pred_type, "Glove embeddings does not have predictions"
        datum = mod_datum_by_preds(args, datum, pred_type)

    # else:
    #     raise Exception('Invalid Datum Modification')
    assert len(datum.index) > 0, "Empty Datum"
    return datum


def read_datum(args, stitch):
    """Load, process, and filter datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        DataFrame: processed and filtered datum
    """
    emb_df = load_datum(args.emb_df_path)
    base_df = load_datum(args.base_df_path)

    df = pd.concat([base_df, emb_df], axis=1)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_datum(args, df, stitch)
    print(f"After processing: Datum now has {len(df)} words")

    df = filter_datum(args, df)
    print(f"After filtering: Datum now has {len(df)} words")

    df = mod_datum(args, df)  # further filter datum based on datum_mod argument

    return df
