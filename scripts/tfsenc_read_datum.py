import os
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle


def load_datum(file_name):
    """Read raw datum from pickle

    Args:
        filename (string): raw datum full file path

    Returns:
        df (df): datum
    """
    datum = load_pickle(file_name)
    df = pd.DataFrame.from_dict(datum)
    if "embeddings" in df.columns:
        df["embeddings"] = df["embeddings"].apply(lambda x: list(x) if x is not None else None)
    return df


def remove_punctuation(df):
    """Remove punctuations from datum"""
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    is_nan = df["embeddings"].apply(lambda x: np.isnan(x).all() if x is not None else True)
    df = df[~is_nan]
    return df


def add_convo_onset_offset(df, stitch_index):
    """Add conversation onset and offset to datum

    Args:
        args (namespace): commandline arguments
        df (df): datum being processed
        stitch_index (list): stitch_index

    Returns:
        df (df): datum with conversation onset and offset
    """
    convo_durations = [
        tuple(stitch_index[x : x + 2]) for x in range(len(stitch_index) - 1)
    ]
    df["convo_onset"], df["convo_offset"] = np.nan, np.nan

    for _, conv in enumerate(df.conversation_id.unique()):
        edges = convo_durations[conv - 1]
        df.loc[df.conversation_id == conv, "convo_onset"] = edges[0]
        df.loc[df.conversation_id == conv, "convo_offset"] = edges[1]

    return df


def mod_datum_arg_parse(arg, mode, default_val="1"):
    """Pare emb_mod argument and get the number of steps to shift/concat"""
    partial = arg[arg.find(mode) + len(mode) :]

    if partial.find("-") >= 0:  # if there is another tag later
        partial = partial[: partial.find("-")]
    else:
        pass
    if len(partial) == 0:  # no number provided
        partial = default_val  # defaults to 1

    step = -1
    if "n" in partial:
        step = 1
        if partial == "n":
            partial = default_val
        else:
            partial = partial[1:]
    assert partial.isdigit()
    shift_num = int(partial)

    return (shift_num, step)


def shift_emb(args, datum, mode="shift-emb"):
    """Shift embeddings based on emb_mod argument

    Args:
        args (namespace): commandline arguments
        datum (df): word datum
        mode (string): concat-emb

    Returns:
        datum (df): word datum with shifted embeddings
    """
    shift_num, step = mod_datum_arg_parse(args.emb_mod, mode)
    print(f"{mode} {shift_num} * {step * -1} steps ")

    before_shift_num = len(datum.index)
    datum2 = datum.copy()  # setting copy to avoid warning
    for i in np.arange(shift_num):
        datum2.loc[:, "embeddings"] = datum2.embeddings.shift(step)
        datum2 = datum2[datum2.conversation_id.shift(step) == datum2.conversation_id]
    datum = datum2  # reassign back to datum
    print(f"Shifting resulted in {before_shift_num - len(datum.index)} less words")

    return datum


def concat_emb(args, datum, mode="concat-emb"):
    """Concatenate embeddings based on emb_mod argument

    Args:
        args (namespace): commandline arguments
        datum (df): word datum
        mode (string): concat-emb

    Returns:
        datum (df): word datum with concatenated embeddings
    """
    shift_num, step = mod_datum_arg_parse(args.emb_mod, mode)
    print(f"{mode} {shift_num} * {step * -1} steps ")

    before_shift_num = len(datum.index)
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings_shifted"] = datum2.embeddings
    for i in np.arange(shift_num):
        datum2.loc[:, "embeddings_shifted"] = datum2.embeddings_shifted.shift(step)
        datum2 = datum2[datum2.conversation_id.shift(step) == datum2.conversation_id]

        def concat(x):
            return np.concatenate((x["embeddings"], x["embeddings_shifted"]))

        datum2.loc[:, "embeddings"] = datum2.apply(concat, axis=1)
    datum = datum2  # reassign back to datum
    print(f"Concatenating resulted in {before_shift_num - len(datum.index)} less words")

    return datum


def zeroshot_datum(df):
    """Sample unique words from datum

    Args:
        datum (df): word datum

    Returns:
        datum (df): zeroshot word datum
    """
    dfz = (
        df[["word", "adjusted_onset"]]
        .groupby("word")
        .apply(lambda x: x.sample(1, random_state=42))
    )
    dfz.reset_index(level=1, inplace=True)
    dfz.sort_values("adjusted_onset", inplace=True)
    df = df.loc[dfz.level_1.values]
    print(f"Zeroshot created datum with {len(df)} words")

    return df


def rand_emb(df):
    """Assign random embeddings to datum

    Args:
        datum (df): word datum

    Returns:
        datum (df): word datum
    """
    emb_max = df.embeddings.apply(max).max()
    emb_min = df.embeddings.apply(min).min()

    rand_emb = np.random.random((len(df), 50))
    rand_emb = rand_emb * (emb_max - emb_min) + emb_min
    df2 = df.copy()  # setting copy to avoid warning
    df2["embeddings"] = list(rand_emb)
    df = df2  # reassign back to datum
    print(f"Generated random embeddings for {len(df)} words")

    return df


def arb_emb(df):
    """Assign arbitrary embeddings to datum (same emb for same word)

    Args:
        datum (df): word datum

    Returns:
        datum (df): word datum
    """
    df2 = zeroshot_datum(df)
    df2 = df2.loc[:, ("word", "embeddings")]
    df2.reset_index(drop=True, inplace=True)
    df2 = rand_emb(df2)
    df = df.drop("embeddings", axis=1, errors="ignore")

    df = df.merge(df2, how="left", on="word")
    df.sort_values(["conversation_id", "index"], inplace=True)
    print(f"Arbitrary embeddings created for {len(df)} words")

    return df


def ave_emb(datum):
    """Average embeddings across tokens

    Args:
        datum (df): datum of tokens

    Returns:
        datum (df): datum of words
    """
    print("Averaging embeddings across tokens")

    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "embeddings"
    ].apply(lambda x: mean_emb(x))
    mean_embs = pd.DataFrame(mean_embs)

    # replace embeddings
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]
    mean_embs.set_index(datum.index, inplace=True)
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings"] = mean_embs.embeddings
    datum = datum2  # reassign back to datum

    return datum


def normalize_embeddings(args, df):
    """Normalize the embeddings
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

    Args:
        args (namespace): arguments
        df (df): word datum

    Returns:
        df (df): word datum
    """
    print("Normalize Embeddings")
    k = np.array(df.embeddings.tolist())

    try:
        k = normalize(k, norm=args.emb_norm, axis=1)
        df2 = df.copy()  # setting copy to avoid warning
        df2["embeddings"] = k.tolist()
        df = df2  # reassign back to datum
    except ValueError:
        print("Error in normalization")

    return df


def trim_datum(args, datum):
    """Trim the datum for lags outside of convo on/offset boundaries

    Args:
        args (namespace): commandline arguments
        datum (df): processed and filtered datum

    Returns:
        datum (df): datum with trimmed words
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


def process_embeddings(args, df):
    """Process the datum embeddings based on input arguments

    Args:
        args (namespace): commandline arguments
        df (df) : raw datum as a DataFrame

    Returns:
        df (df): processed datum with correct embeddings
    """
    # drop NaN / None embeddings
    if args.emb == "glove50":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)

    # Embedding manipulation (token level)
    if "shift-emb" in args.emb_mod:  # shift embeddings
        df = shift_emb(args, df, "shift-emb")
    elif "concat-emb" in args.emb_mod:  # concatenate embeddings
        df = concat_emb(args, df, "concat-emb")

    # Average embeddings across tokens per word
    if "glove" not in args.emb:
        emb_name_subset = args.emb.split("_")[0]
        if df[f"{emb_name_subset}_token_is_root"].sum() < len(df):
            df = ave_emb(df)  # average embs per word

    # Embedding manipulation (word level)
    if "rand" in args.emb_mod:  # random embeddings
        df = rand_emb(df)
    elif "arb" in args.emb_mod:  # artibtrary embeddings
        df = arb_emb(df)
    else:
        pass

    # Normalize embeddings
    if args.emb_norm:
        df = normalize_embeddings(args, df)

    return df


def filter_datum(args, df):
    """Process/clean/filter datum based on args

    Args:
        args (namespace): commandline arguments
        df (df): processed datum

    Returns:
        df (df): filtered datum
    """

    # create mask for further filtering
    common = np.repeat(True, len(df))

    # get rid of tokens without onset/offset
    common &= df.adjusted_onset.notna()
    common &= df.adjusted_offset.notna()
    common &= df.onset.notna()
    common &= df.offset.notna()

    # get rid of tokens without proper speaker
    speaker_mask = df.speaker.str.contains("Speaker")  # TODO
    common &= speaker_mask

    # filter based on arguments: nonwords, word_freq
    if args.exclude_nonwords:
        common &= ~df.is_nonword

    freq_mask = df.word_freq_overall >= args.minimum_word_frequency
    common &= freq_mask

    df = df[common]

    return df


def process_datum(args, df, stitch):
    """Process datum, including

    Args:
        args (namespace): commandline arguments
        df: processed datum
        stitch: stitch index

    Returns:
        DataFrame: processed datum with correct conversations
    """
    # Conversation level
    df = add_convo_onset_offset(df, stitch)  # add conversation onset/offset
    df = df.loc[df.conversation_id.isin(args.conv_ids)]  # Select conversations

    # Token/word level
    df = process_embeddings(args, df)  # manipulate embeddings
    df = filter_datum(args, df)  # filter datum
    if args.trim_conv_edges:  # trim edges (lags outside of convo on/offset)
        df = trim_datum(args, df)

    assert len(df.index) > 0, "Empty Datum"

    return df


def read_datum(args, stitch):
    """Load and process datum

    Args:
        args (namespace): commandline arguments
        stitch (list): stitch_index

    Returns:
        df (df): processed and filtered datum
    """
    args.kernel_sizes = []
    for idx, emb_df_path in enumerate(args.emb_df_paths.split()):
        curr_emb_df = load_datum(emb_df_path)
        args.kernel_sizes.append(len(curr_emb_df['embeddings'].iloc[1]))
        if idx == 0:
            emb_df = curr_emb_df
        else:
            assert(emb_df.shape[0] == curr_emb_df.shape[0])
            emb_df['embeddings'] = emb_df['embeddings'] + curr_emb_df['embeddings']
    base_df = load_datum(args.base_df_path)
    if "refrecur" in args.emb:
        base_df = base_df[base_df["annot_type"].isin(["ref", "recur"])]
        base_df.index = np.arange(len(base_df))
        assert(base_df.shape[0] == emb_df.shape[0]), \
            f"Base df and emb df length mismatch: {base_df.shape[0]} vs {emb_df.shape[0]}"
    df = pd.merge(base_df, emb_df, left_index=True, right_index=True)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_datum(args, df, stitch)
    print(f"Datum final length: {len(df)}")

    return df
