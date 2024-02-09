import os
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle

# import gensim.downloader as api
# import re


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    is_nan = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~is_nan]

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


def shift_emb(args, datum, mode="shift-emb"):
    """Shift the embeddings based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        mode: concat-emb

    Returns:
        DataFrame: datum with shifted embeddings
    """
    shift_num, step = mod_datum_arg_parse(args.emb_mod, mode)
    print(f"{mode} {shift_num} * {step * -1} steps ")

    before_shift_num = len(datum.index)
    datum2 = datum.copy()  # setting copy to avoid warning
    for i in np.arange(shift_num):
        datum2.loc[:, "embeddings"] = datum2.embeddings.shift(step)
        if (
            "blenderbot-small" in args.emb_type.lower()
            or "bert" in args.emb_type.lower()
        ):
            datum2 = datum2[
                (
                    datum2.production.shift(step) == datum2.production
                    and datum2.conversation_id.shift(step) == datum2.conversation_id
                )
            ]
        else:
            datum2 = datum2[
                datum2.conversation_id.shift(step) == datum2.conversation_id
            ]
    datum = datum2  # reassign back to datum
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
    shift_num, step = mod_datum_arg_parse(args.emb_mod, mode)
    print(f"{mode} {shift_num} * {step * -1} steps ")

    before_shift_num = len(datum.index)
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings_shifted"] = datum2.embeddings
    for i in np.arange(shift_num):
        datum2.loc[:, "embeddings_shifted"] = datum2.embeddings_shifted.shift(step)
        if (
            "blenderbot-small" in args.emb_type.lower()
            or "bert" in args.emb_type.lower()
        ):
            datum2 = datum2[
                (
                    datum2.production.shift(step) == datum2.production
                    and datum2.conversation_id.shift(step) == datum2.conversation_id
                )
            ]
        else:
            datum2 = datum2[
                datum2.conversation_id.shift(step) == datum2.conversation_id
            ]

        def concat(x):
            return np.concatenate((x["embeddings"], x["embeddings_shifted"]))

        datum2.loc[:, "embeddings"] = datum2.apply(concat, axis=1)
    datum = datum2  # reassign back to datum
    print(f"Concatenating resulted in {before_shift_num - len(datum.index)} less words")

    return datum


def rand_emb(df):
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
    df2 = zeroshot_datum(df)
    df2 = df2.loc[:, ("word", "embeddings")]
    df2.reset_index(drop=True, inplace=True)
    df2 = rand_emb(df2)
    df = df.drop("embeddings", axis=1, errors="ignore")

    df = df.merge(df2, how="left", on="word")
    df.sort_values(["conversation_id", "index"], inplace=True)
    print(f"Arbitrary embeddings created for {len(df)} words")

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
    print("Normalize Embeddings")
    k = np.array(df.embeddings.tolist())

    try:
        k = normalize(k, norm=args.normalize, axis=1)
        df2 = df.copy()  # setting copy to avoid warning
        df2["embeddings"] = k.tolist()
        df = df2  # reassign back to datum
    except ValueError:
        print("Error in normalization")

    return df


def ave_emb(datum):
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


def zeroshot_datum(df):
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


def load_glove_embeddings(args):
    glove_base_df_path = os.path.join(
        args.PICKLE_DIR, "embeddings", "glove50", "full", "base_df.pkl"
    )
    glove_emb_df_path = os.path.join(
        args.PICKLE_DIR,
        "embeddings",
        "glove50",
        "full",
        "cnxt_0001",
        "layer_01.pkl",
    )

    glove_base_df = load_datum(glove_base_df_path)
    glove_emb_df = load_datum(glove_emb_df_path)
    glove_df = pd.merge(glove_base_df, glove_emb_df, left_index=True, right_index=True)
    glove_df = glove_df[glove_df[f"in_{args.emb_type}"]]
    glove_df = glove_df.loc[:, ["adjusted_onset", "word", "embeddings"]]

    return glove_df


def process_embeddings(args, df):
    """Process the datum embeddings based on input arguments

    Args:
        args (namespace): commandline arguments
        df : raw datum as a DataFrame

    Returns:
        DataFrame: processed datum with correct embeddings
    """

    # drop NaN / None embeddings
    if args.emb_type == "glove50":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)

    # add prediction embeddings (force to glove)
    if "glove" in args.emb_mod:
        mask = df[f"in_glove50"] & df[f"{args.emb_type}_token_is_root"]
        df = df[mask]
        df.drop(
            ["embeddings"],
            axis=1,
            errors="ignore",
            inplace=True,
        )  # delete current embeddings

        glove_df = load_glove_embeddings(args)
        df = df[df.adjusted_onset.notna()]
        glove_df = glove_df[glove_df.adjusted_onset.notna()]
        breakpoint()
        df = df.merge(glove_df, how="inner", on=["adjusted_onset", "word"])

    # Embedding manipulation
    if "shift-emb" in args.emb_mod:  # shift embeddings
        df = shift_emb(args, df, "shift-emb")
    elif "concat-emb" in args.emb_mod:  # concatenate embeddings
        df = concat_emb(args, df, "concat-emb")
    elif "-rand" in args.emb_mod:  # random embeddings
        df = rand_emb(df)
    elif "-arb" in args.emb_mod:  # artibtrary embeddings
        df = arb_emb(df)
    else:
        pass

    return df


def process_conversations(args, df, stitch):
    """Select conversations for the datum

    Args:
        args (namespace): commandline arguments
        df: processed datum
        stitch: stitch index

    Returns:
        DataFrame: processed datum with correct conversations
    """
    # filter bad convos (specifically for 676)
    df = df.loc[~df["conversation_id"].isin(args.bad_convos)]
    assert len(stitch) - len(args.bad_convos) == df.conversation_id.nunique() + 1

    # add conversation onset/offset (should not need later)
    df = add_convo_onset_offset(args, df, stitch)

    # Single convo
    if args.conversation_id:  # picking single conversation
        datum = datum[datum.conversation_id == args.conversation_id]
        datum.convo_offset = datum["convo_offset"] - datum["convo_onset"]
        datum.convo_onset = 0
        print(f"Running conversation {args.conversation_id} with {len(datum)} words")
    return df


def filter_datum(args, df):
    """Process/clean/filter datum based on args

    Args:
        args (namespace): commandline arguments
        df: processed datum
        stitch: stitch index

    Returns:
        DataFrame: filtered datum
    """

    ## Trimming datum
    if "notrim" in args.datum_mod:  # no need for edge trimming
        pass
    else:
        df = trim_datum(args, df)  # trim edges

    # create mask for further filtering
    common = np.repeat(True, len(df))

    # get rid of tokens without onset/offset
    common &= df.adjusted_onset.notna()
    common &= df.adjusted_offset.notna()
    common &= df.onset.notna()
    common &= df.offset.notna()

    # get rid of tokens without proper speaker
    speaker_mask = df.speaker.str.contains("Speaker")
    common &= speaker_mask

    # filter based on arguments: nonwords, word_freq
    if args.exclude_nonwords:
        common &= ~df.is_nonword

    freq_mask = df.word_freq_overall >= args.min_word_freq
    common &= freq_mask

    # filter based on align with arguments
    for model in args.align_with:
        if (
            model == "glove50" and args.emb_type != "glove50"
        ):  # when aligning with glove
            common = (
                common & df[f"{args.emb_type}_token_is_root"]
            )  # also ensure word=token
        print(f"Aligning with {model}")
        common = common & df[f"in_{model}"]

    df = df[common]

    return df


def mod_datum_by_preds(args, datum):
    """Filter the datum based on the predictions of a potentially different model

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        emb_type: embedding type needed to filter the datum

    Returns:
        DataFrame: further filtered datum
    """

    print(f"Using {args.emb_type} predictions")

    # modify datum based on correct or incorrect predictions
    if "incorrect" in args.datum_mod:  # select words predicted incorrectly
        rank, _ = mod_datum_arg_parse(args.datum_mod, "incorrect", "5")
        datum = datum[datum.true_pred_rank > rank]  # incorrect
        print(f"Selected {len(datum.index)} top{rank} incorrect words")
    elif "correct" in args.datum_mod:  # select words predicted correctly
        rank, _ = mod_datum_arg_parse(args.datum_mod, "correct", "5")
        datum = datum[datum.true_pred_rank <= rank]  # correct
        print(f"Selected {len(datum.index)} top{rank} correct words")
    elif "improb" in args.datum_mod:  # select low pred_prob words
        percentile, _ = mod_datum_arg_parse(args.datum_mod, "improb", "30")
        bot = datum.true_pred_prob.quantile(percentile / 100)
        datum = datum[datum.true_pred_prob <= bot]
        print(f"Selected {len(datum.index)} bot pred prob words")
    elif "prob" in args.datum_mod:  # select high pred_prob words
        percentile, _ = mod_datum_arg_parse(args.datum_mod, "prob", "30")
        top = datum.true_pred_prob.quantile(1 - percentile / 100)
        datum = datum[datum.true_pred_prob >= top]
        print(f"Selected {len(datum.index)} top pred prob words")

    # elif args.datum_mod == emb_type + "-pred": # for incorrectly predicted words, replace with top 1 pred (only used for podcast glove)
    #     glove = api.load('glove-wiki-gigaword-50')
    #     datum['embeddings'] = datum.top1_pred.str.strip().apply(lambda x: get_vector(x.lower(), glove))
    #     datum = datum[datum.embeddings.notna()]
    #     print(f'Changed words into {emb_type} top predictions')
    else:  # exception
        raise Exception("Invalid Datum Modification")

    return datum


def mod_datum_arg_parse(arg, mode, default_val="1"):
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


def mod_datum(args, datum):
    """Filter the datum based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: further filtered datum
    """

    ## Token manipulation
    if "-all" in args.datum_mod:  # all tokens
        pass

    elif "-zeroshot" in args.datum_mod:  # zeroshot tokens
        datum = zeroshot_datum(datum)

    else:  # modify datum based on predictions
        datum = mod_datum_by_preds(args, datum)

    # else:
    #     raise Exception('Invalid Datum Modification')

    # Average Embeddings per word
    if "glove" not in args.emb_type and "glove" not in args.emb_mod:
        if datum[f"{args.emb_type}_token_is_root"].sum() < len(datum):
            datum = ave_emb(datum)  # average embs per word

    # Normalize Embeddings
    if args.normalize:
        datum = normalize_embeddings(args, datum)

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

    df = pd.merge(
        base_df, emb_df, left_index=True, right_index=True
    )  # TODO Needs testing (either bert_utterance or whisper)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_conversations(args, df, stitch)
    df = process_embeddings(args, df)
    print(f"After processing: Datum now has {len(df)} words")

    df = filter_datum(args, df)
    print(f"After filtering: Datum now has {len(df)} words")

    df = mod_datum(args, df)  # further filter datum based on datum_mod argument
    print(f"Datum final length: {len(df)}")

    return df
