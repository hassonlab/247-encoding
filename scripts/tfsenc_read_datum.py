import os
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from utils import load_pickle

# import gensim.downloader as api
# import re


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding"""
    df["is_nan"] = df["embeddings"].apply(lambda x: np.isnan(x).all())
    df = df[~df["is_nan"]]

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
        df["embeddings"] = k.tolist()
    except ValueError:
        raise Exception("Error in normalization")

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

    df = df.loc[~df["conversation_id"].isin(args.bad_convos)]  # filter bad convos
    assert len(stitch) - len(args.bad_convos) == df.conversation_id.nunique() + 1

    df = df[df.adjusted_onset.notna()]
    df = add_convo_onset_offset(args, df, stitch)

    if args.emb_type == "glove50":
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)

    # df = df[~df['glove50_embeddings'].isna()]
    # if encoding is on glove embeddings copy them into 'embeddings' column
    # if args.emb_type == "glove50":
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
        if model == "glove50":
            model = "glove"  ## HACK
            if args.emb_type != "glove50":  # when aligning with glove
                common = (
                    common & df[f"{args.emb_type}_token_is_root"]
                )  # also ensure word=token
        print(f"Aligning with {model}")
        common = common & df[f"in_{model}"]

    if args.exclude_nonwords:  # filter based on exclude_nonwords argument
        common &= ~df.is_nonword

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
    if emb_type in args.emb_df_path:  # current datum has the correct emb_type
        pass
    else:  # current datum does not have the correct emb_type, need to load a second datum
        # load second datum
        if emb_type == "gpt2-xl":
            second_base_df_path = os.path.join(
                args.PICKLE_DIR, "embeddings", "gpt2-xl", "full", "base_df.pkl"
            )
            second_emb_df_path = os.path.join(
                args.PICKLE_DIR,
                "embeddings",
                "gpt2-xl",
                "full",
                "cnxt_1024",
                "layer_48.pkl",
            )
        else:
            raise Exception("Not implemented")  # TODO

        second_base_df = load_datum(second_base_df_path)
        second_emb_df = load_datum(second_emb_df_path)

        second_datum = pd.merge(
            second_base_df, second_emb_df, left_index=True, right_index=True
        )
        # second_base_df.reset_index(
        #     drop=True, inplace=True
        # )  # so concatenate can be aligned correctly
        # second_datum = pd.concat([second_base_df, second_emb_df], axis=1)
        if args.emb_type == "glove50":
            second_datum = second_datum[
                second_datum["gpt2-xl_token_is_root"] & second_datum["in_glove50"]
            ]
        second_datum = second_datum.loc[
            :,
            [
                "adjusted_onset",
                "word",
                "top1_pred",
                "top1_pred_prob",
                "true_pred_prob",
                "true_pred_rank",
            ],
        ]

        # merge second datum prediction columns to datum
        datum = datum.drop(
            ["top1_pred", "top1_pred_prob", "true_pred_prob", "true_pred_rank"],
            axis=1,
            errors="ignore",
        )  # delete the current top predictions if any
        datum = datum[datum.adjusted_onset.notna()]
        second_datum = second_datum[second_datum.adjusted_onset.notna()]
        datum = datum.merge(second_datum, how="inner", on=["adjusted_onset", "word"])
    print(f"Using {emb_type} predictions")

    # modify datum based on correct or incorrect predictions
    if "incorrect" in args.datum_mod:  # select words predicted incorrectly
        rank, _ = mod_datum_arg_parse(args, "incorrect", "5")
        datum = datum[datum.true_pred_rank > rank]  # incorrect
        print(f"Selected {len(datum.index)} top{rank} incorrect words")
    elif "correct" in args.datum_mod:  # select words predicted correctly
        rank, _ = mod_datum_arg_parse(args, "correct", "5")
        datum = datum[datum.true_pred_rank <= rank]  # correct
        print(f"Selected {len(datum.index)} top{rank} correct words")
    elif "improb" in args.datum_mod:  # select low pred_prob words
        percentile, _ = mod_datum_arg_parse(args, "improb", "30")
        bot = datum.true_pred_prob.quantile(percentile / 100)
        datum = datum[datum.true_pred_prob <= bot]
        print(f"Selected {len(datum.index)} bot pred prob words")
    elif "prob" in args.datum_mod:  # select high pred_prob words
        percentile, _ = mod_datum_arg_parse(args, "prob", "30")
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


def mod_datum_arg_parse(args, mode, default_val="1"):
    partial = args.datum_mod[args.datum_mod.find(mode) + len(mode) :]

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
    """Shift the embeddings based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        mode: concat-emb

    Returns:
        DataFrame: datum with shifted embeddings
    """
    shift_num, step = mod_datum_arg_parse(args, mode)
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
    shift_num, step = mod_datum_arg_parse(args, mode)
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


def mod_datum(args, datum):
    """Filter the datum based on datum_mod argument

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum

    Returns:
        DataFrame: further filtered datum
    """
    ## Trimming datum
    if "notrim" in args.datum_mod:  # no need for edge trimming
        pass
    else:
        datum = trim_datum(args, datum)  # trim edges

    ## Single convo
    if args.conversation_id:  # picking single conversation
        datum = datum[datum.conversation_id == args.conversation_id]
        datum.convo_offset = datum["convo_offset"] - datum["convo_onset"]
        datum.convo_onset = 0
        print(f"Running conversation {args.conversation_id} with {len(datum)} words")

    ## Embedding manipulation
    if "shift-emb" in args.datum_mod:  # shift embeddings
        datum = shift_emb(args, datum, "shift-emb")
    elif "concat-emb" in args.datum_mod:  # concatenate embeddings
        datum = concat_emb(args, datum, "concat-emb")
    elif "-rand" in args.datum_mod:  # random embeddings
        datum = rand_emb(datum)
    elif "-arb" in args.datum_mod:  # artibtrary embeddings
        datum = arb_emb(datum)
    else:
        pass

    if "glove" not in args.emb_type and "glove50" not in args.align_with:
        datum = ave_emb(datum)  # average embs per word

    ## Token manipulation
    if "-all" in args.datum_mod:  # all tokens
        pass

    elif "-zeroshot" in args.datum_mod:  # zeroshot tokens
        datum = zeroshot_datum(datum)

    else:  # modify datum based on predictions
        pred_type = args.emb_type
        if "gpt2-xl" in args.datum_mod:  # if prediction from a different model
            pred_type = "gpt2-xl"
        elif "blenerbot-small" in args.datum_mod:
            pred_type = "blenderbot-small"
        assert "glove" not in pred_type, "Glove embeddings does not have predictions"
        datum = mod_datum_by_preds(args, datum, pred_type)

    # else:
    #     raise Exception('Invalid Datum Modification')

    assert len(datum.index) > 0, "Empty Datum"
    return datum


def run_pca(args, df):
    pca_to = args.pca_to
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    assert embs.shape[1] % 10 == 0, "Something wrong with emb shape"

    if "-" in args.window_num:  # range
        start_win = args.window_num[: args.window_num.find("-")]
        end_win = args.window_num[args.window_num.find("-") + 1 :]
    else:  # onewindow
        start_win = end_win = args.window_num
    assert start_win.isdigit()
    assert end_win.isdigit()
    emb_dim = 384  ## HACK whisper-tiny.en
    # emb_dim = 4096  ## HACK whisper-medium.en
    # emb_dim = 6400  ## HACK whisper-large-v2

    if "full-en-offset" in args.base_df_path:
        print(f"Taking win {start_win} to {end_win} from the back")
        start_idx = int(end_win) * emb_dim * -1
        end_idx = (int(start_win) - 1) * emb_dim * -1
        if start_win == "1":
            embs = embs[:, start_idx:]
        else:
            embs = embs[:, start_idx:end_idx]
    else:
        print(f"Taking win {start_win} to {end_win}")
        start_idx = (int(start_win) - 1) * emb_dim
        end_idx = int(end_win) * emb_dim
        embs = embs[:, start_idx:end_idx]

    print(f"PCA from {embs.shape[1]} to {pca_to}")
    print(f"PCA explained variance: {sum(pca.explained_variance_)}")
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df


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

    if "whisper" in args.emb_type:  ## HACK
        base_df = base_df.dropna(subset=["onset", "offset"])
        base_df.reset_index(drop=True, inplace=True)
        assert len(base_df) == len(emb_df)

    df = pd.merge(
        base_df, emb_df, left_index=True, right_index=True
    )  # TODO Needs testing (either bert_utterance or whisper)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_datum(args, df, stitch)
    print(f"After processing: Datum now has {len(df)} words")

    df = filter_datum(args, df)
    print(f"After filtering: Datum now has {len(df)} words")

    df = mod_datum(args, df)  # further filter datum based on datum_mod argument
    print(f"Datum final length: {len(df)}")

    # if args.project_id == "tfs":  # HACK for early pca for efficiency
    #     print(f"Running early pca for large embeddings")
    #     df = run_pca(args, df)

    return df
