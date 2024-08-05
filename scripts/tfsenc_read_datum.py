import os
import string

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from utils import load_pickle, save_pickle
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

import gensim.downloader as api

# import re

import nltk


def get_pos(df):
    words_orig, part_of_speech = zip(*nltk.pos_tag(df.word, tagset="universal"))
    df = df.assign(part_of_speech=part_of_speech)
    return df


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def get_vectors(x, glove, length):
    preds = "".join(x)  # join predictions
    preds = preds.lower()  # lowercase
    preds = preds.translate(
        str.maketrans("", "", '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')
    )  # remove all punc besides single quote
    preds = preds.split()  # split into words

    emb = np.array([])
    for pred in preds:
        if len(emb) == length * 50:
            return emb
        try:
            emb = np.append(emb, glove.get_vector(pred))
        except KeyError:
            pass

    return None


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
    # datum2.loc[:, "pred_shifted"] = datum2.true_pred_prob
    # datum2.loc[:, "rank_shifted"] = datum2.true_pred_rank
    for i in np.arange(shift_num):
        datum2.loc[:, "embeddings_shifted"] = datum2.embeddings_shifted.shift(step)
        # datum2.loc[:, "pred_shifted"] = datum2.pred_shifted.shift(step)
        # datum2.loc[:, "rank_shifted"] = datum2.rank_shifted.shift(step)
        # datum2 = datum2[ # split prob comp concatenation
        #     datum2.production.shift(step).eq(datum2.production)
        #     & datum2.conversation_id.shift(step).eq(datum2.conversation_id)
        # ]
        print("NOT CARE FOR COMP/PROD SEPARATION")
        datum2 = datum2[datum2.conversation_id.shift(step) == datum2.conversation_id]

        def concat(x):
            return np.concatenate((x["embeddings"], x["embeddings_shifted"]))

        datum2.loc[:, "embeddings"] = datum2.apply(concat, axis=1)
        # datum2.loc[:, "true_pred_prob"] = (
        #     datum2.true_pred_prob + datum2.pred_shifted
        # )
        # datum2.loc[:, "true_pred_rank"] = (
        #     datum2.true_pred_rank + datum2.rank_shifted
        # )

    # datum2.loc[:, "true_pred_prob"] = datum2.true_pred_prob / (shift_num + 1)
    # datum2.loc[:, "true_pred_rank"] = datum2.true_pred_rank / (shift_num + 1)

    datum = datum2  # reassign back to datum
    print(f"Concatenating resulted in {before_shift_num - len(datum.index)} less words")

    return datum


def concat_emb_improb(args, datum, mode="concat-emb-improb"):
    """Concatenate the embeddings based on datum_mod argument, but concatenate
        only the next improb words

    Args:
        args (namespace): commandline arguments
        datum: processed and filtered datum
        mode: concat-emb

    Returns:
        DataFrame: datum with shifted embeddings
    """
    if "concat-emb-improb" in args.emb_mod:
        concat_shift_num = 0
        shift_num, step = mod_datum_arg_parse(args.emb_mod, mode)
        print(f"{mode} {shift_num} * {step * -1} steps ")
    else:
        concat_shift_num, shift_step = mod_datum_arg_parse(args.emb_mod, "concat-emb")
        print(f"Concat Consecutive {concat_shift_num} * {shift_step * -1} steps ")
        shift_num, step = mod_datum_arg_parse(args.emb_mod, "improb")
        print(f"Concat Improb {shift_num} * {step * -1} steps ")

    before_shift_num = len(datum.index)
    bot = datum.true_pred_prob.quantile(0.3)
    datum2 = datum.copy()  # setting copy to avoid warning

    for i in np.arange(shift_num):
        # get next improb embeddings
        if "level_0" in datum2.columns:
            idx_col = "level_0"
        else:
            idx_col = "index"

        datum2["improb"] = np.nan
        datum2.loc[datum2.true_pred_prob.lt(bot), "improb"] = datum2[idx_col]
        datum2.improb = (
            datum2.improb.bfill().shift(-1 - concat_shift_num).ffill().astype(int)
        )

        new_embs = datum2.loc[datum2.improb, "embeddings"]
        new_embs.index = datum2.index
        datum2["embeddings_next"] = new_embs

        # get rid of improb from future conversations
        datum2["improb_check"] = np.nan
        datum2.loc[datum2.true_pred_prob.lt(bot), "improb_check"] = datum2[
            "conversation_id"
        ]
        datum2.improb_check = (
            datum2.improb_check.bfill().shift(-1 - concat_shift_num).ffill().astype(int)
        )

        # for each convo, take only words before the last improb word
        datum2 = datum2[
            datum2.improb.gt(datum2[idx_col] + concat_shift_num)
            & datum2.improb_check.eq(datum2.conversation_id)
        ]

        # concat embeddings
        def concat(x):
            return np.concatenate((x["embeddings"], x["embeddings_next"]))

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


def ave_emb_preds(datum):
    print("Averaging embeddings and preds across tokens")

    # calculate mean embeddings
    def mean_emb(embs):
        return np.array(embs.values.tolist()).mean(axis=0).tolist()

    mean_embs = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "embeddings"
    ].apply(lambda x: mean_emb(x))
    mean_embs = pd.DataFrame(mean_embs)
    mean_preds = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "true_pred_prob"
    ].apply(lambda x: mean_emb(x))
    mean_preds = pd.DataFrame(mean_preds)

    # replace embeddings
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]
    mean_embs.set_index(datum.index, inplace=True)
    mean_preds.set_index(datum.index, inplace=True)
    datum2 = datum.copy()  # setting copy to avoid warning
    datum2.loc[:, "embeddings"] = mean_embs.embeddings
    datum2.loc[:, "true_pred_prob"] = mean_preds.true_pred_prob
    datum = datum2  # reassign back to datum

    return datum


def ave_pred(datum):
    print("Averaging predictions across tokens")

    datum["true_pred_prob"] = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "true_pred_prob"
    ].transform(np.mean)
    datum["true_pred_rank"] = datum.groupby(["adjusted_onset", "word"], sort=False)[
        "true_pred_rank"
    ].transform(np.mean)

    # taking only first instance
    idx = (
        datum.groupby(["adjusted_onset", "word"], sort=False)["token_idx"].transform(
            min
        )
        == datum["token_idx"]
    )
    datum = datum[idx]

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
    if len(glove_base_df) != len(glove_emb_df):  # HACK old
        glove_df = pd.merge(
            glove_base_df, glove_emb_df, left_index=True, right_index=True
        )
    else:
        glove_base_df.reset_index(drop=False, inplace=True)
        glove_df = pd.concat([glove_base_df, glove_emb_df], axis=1)
    glove_df = glove_df[glove_df[f"in_{args.emb_type}"]]  # HACK 2 turn of this line
    glove_df = glove_df.loc[:, ["adjusted_onset", "word", "embeddings"]]

    return glove_df


def load_glove_embeddings2(args):
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
    if len(glove_base_df) != len(glove_emb_df):  # HACK old
        glove_df = pd.merge(
            glove_base_df, glove_emb_df, left_index=True, right_index=True
        )
    else:
        glove_base_df.reset_index(drop=False, inplace=True)
        glove_df = pd.concat([glove_base_df, glove_emb_df], axis=1)
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

    if "glove50-predict" in args.emb_mod:  # predictions
        print("Glove embeddings with gpt2 predictions")
        args.emb_type = "glove50"
        glove = api.load("glove-wiki-gigaword-50")
        df = df.dropna(subset=["top1_pred"])
        emb_concat_len = 1
        if "glove50-predict5" in args.emb_mod:
            emb_concat_len = 5
        df.loc[:, "embeddings"] = df.top1_pred.apply(
            lambda x: get_vectors(x, glove, emb_concat_len)
        )
        df = df[df.embeddings.notna()]
        args.emb_mod = ""
        df.loc[:, "true_pred_prob"] = df.true_pred_prob.apply(lambda x: (x[0]))
        df.drop_duplicates(subset=["adjusted_onset", "word"], inplace=True)

    # drop NaN / None embeddings
    if "glove50" in args.emb_type:
        df = df.dropna(subset=["embeddings"])
    else:
        df = drop_nan_embeddings(df)
        df = remove_punctuation(df)
    # add prediction embeddings (force to glove)
    if "glove50-concat-emb" in args.emb_mod:  # glove concat
        df.drop(
            ["embeddings"], axis=1, errors="ignore", inplace=True
        )  # delete current embeddings
        df = ave_pred(df)
        glove_df = load_glove_embeddings2(args)
        df = df[df.adjusted_onset.notna()]
        glove_df = glove_df[glove_df.adjusted_onset.notna()]
        df = df.merge(glove_df, how="inner", on=["adjusted_onset", "word"])
    elif "glove50" in args.emb_mod:
        # HACK 2: turn the following two lines off
        mask = df[f"in_glove50"] & df[f"{args.emb_type}_token_is_root"]
        df = df[mask]
        df.drop(
            ["embeddings"], axis=1, errors="ignore", inplace=True
        )  # delete current embeddings
        glove_df = load_glove_embeddings(args)
        df = df[df.adjusted_onset.notna()]
        glove_df = glove_df[glove_df.adjusted_onset.notna()]
        df = df.merge(glove_df, how="inner", on=["adjusted_onset", "word"])
    # elif "llama3" in args.emb_mod:
    #     mask = df[f"in_glove50"] & df[f"{args.emb_type}_token_is_root"]
    #     df = df[mask]
    #     df.drop(
    #         ["embeddings"], axis=1, errors="ignore", inplace=True
    #     )  # delete current embeddings

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "meta-llama/Meta-Llama-3-8B",
    #         add_prefix_space=True,
    #         cache_dir=os.path.join(os.path.dirname(os.getcwd()), ".cache"),
    #         local_files_only=True,
    #     )
    #     df["token"] = df.word.apply(tokenizer.tokenize)
    #     df = df.explode("token", ignore_index=False)
    #     df["token_id"] = df["token"].apply(tokenizer.convert_tokens_to_ids)
    #     df["token_idx"] = df.groupby(["adjusted_onset", "word"]).cumcount()
    #     llama_emb = pd.read_pickle("../247-pickling/llama3_dict.pkl")
    #     df = df.merge(llama_emb, how="left", on="token_id")
    #     df = ave_emb(df)
    elif "-pred" in args.emb_mod:
        # df.drop(
        #     ["embeddings"], axis=1, errors="ignore", inplace=True
        # )  # delete current embeddings
        if "llama2-pred" in args.emb_mod:
            hf_model = "meta-llama/Llama-2-7b-hf"
        elif "llama3-pred" in args.emb_mod:
            hf_model = "meta-llama/Meta-Llama-3-8B"

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model,
            add_prefix_space=True,
            cache_dir=os.path.join(os.path.dirname(os.getcwd()), ".cache"),
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            output_hidden_states=True,
            cache_dir=os.path.join(os.path.dirname(os.getcwd()), ".cache"),
            local_files_only=True,
        )

        def get_pred_token_id(x):
            try:
                token_id = tokenizer.convert_tokens_to_ids(x[0])
                assert token_id is not None
                assert token_id != 0
            except:
                token_id = tokenizer.encode(x[0])[-1]
            return token_id

        df["pred_id"] = df.topk_pred.apply(lambda x: get_pred_token_id(x))

        tokens = torch.tensor([df.pred_id.tolist()])
        df["embeddings2"] = model.model.embed_tokens(tokens)[0].tolist()

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

    # Only for Prob-improb
    # common = (
    #     common & df[f"{args.emb_type}_token_is_root"]
    # )  # HACK 1, turn off for #HACK 2
    # common = common & df[f"in_glove50"]

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
    if "incorrect" in args.datum_mod:  # select words predicted incorrectly
        rank, _ = mod_datum_arg_parse(args.datum_mod, "incorrect", "5")
        datum = datum[datum.true_pred_rank > rank]  # incorrect
        print(f"Selected {len(datum.index)} top{rank} incorrect words")
    elif "correct" in args.datum_mod:  # select words predicted correctly
        rank, _ = mod_datum_arg_parse(args.datum_mod, "correct", "5")
        datum = datum[datum.true_pred_rank <= rank]  # correct
        print(f"Selected {len(datum.index)} top{rank} correct words")
    elif "cp" in args.datum_mod and "prob" in args.datum_mod:
        percentile, _ = mod_datum_arg_parse(args.datum_mod, "prob", "30")
        datum_comp = datum[datum.production == 0]
        datum_prod = datum[datum.production == 1]

        top_comp = datum_comp.true_pred_prob.quantile(1 - percentile / 100)
        bot_comp = datum_comp.true_pred_prob.quantile(percentile / 100)
        top_prod = datum_prod.true_pred_prob.quantile(1 - percentile / 100)
        bot_prod = datum_prod.true_pred_prob.quantile(percentile / 100)

        datum_top = datum.loc[
            (
                ((datum.true_pred_prob >= top_comp) & (datum.production == 0))
                | ((datum.true_pred_prob >= top_prod) & (datum.production == 1))
            ),
            :,
        ].copy()
        datum_bot = datum.loc[
            (
                ((datum.true_pred_prob <= bot_comp) & (datum.production == 0))
                | ((datum.true_pred_prob <= bot_prod) & (datum.production == 1))
            ),
            :,
        ].copy()
        if "cp-prob" in args.datum_mod:
            datum = datum_top
            print(f"Selected {len(datum.index)} top pred prob words")
        elif "cp-improb" in args.datum_mod:
            datum = datum_bot
            print(f"Selected {len(datum.index)} bot pred prob words")
        # elif "cp-aligned-prob" in args.datum_mod:
        #     datum = datum_top[datum_top.word.isin(datum_bot.word.unique())]
        #     print(f"Selected {len(datum.index)} top pred prob words")
        # elif "cp-aligned-improb" in args.datum_mod:
        #     datum = datum_bot[datum_bot.word.isin(datum_top.word.unique())]
        #     print(f"Selected {len(datum.index)} bot pred prob words")

    elif "prob" in args.datum_mod:  # select low pred_prob words
        percentile, _ = mod_datum_arg_parse(args.datum_mod, "prob", "30")
        top = datum.true_pred_prob.quantile(1 - percentile / 100)
        bot = datum.true_pred_prob.quantile(percentile / 100)
        datum["word_gap"] = datum.adjusted_onset - datum.adjusted_offset.shift()
        datum_top = datum[datum.true_pred_prob >= top].copy()
        datum_bot = datum[datum.true_pred_prob <= bot].copy()

        # if True:
        #     datum.drop(columns="embeddings", inplace=True)
        #     # datum_top_aligned = datum_top[datum_top.word.isin(datum_bot.word.unique())]
        #     # datum_bot_aligned = datum_bot[datum_bot.word.isin(datum_top.word.unique())]
        #     datum_mid = datum[
        #         datum.true_pred_prob.gt(bot) & datum.true_pred_prob.lt(top)
        #     ].copy()
        #     datum_top.to_pickle(f"{args.sid}_llama2_32_prob.pkl")
        #     datum_bot.to_pickle(f"{args.sid}_llama2_32_improb.pkl")
        #     # datum_top_aligned.to_pickle(f"{args.sid}_gpt2_32_prob_a.pkl")
        #     # datum_bot_aligned.to_pickle(f"{args.sid}_gpt2_32_improb_a.pkl")
        #     datum_mid.to_pickle(f"{args.sid}_llama2_32_mid.pkl")
        #     breakpoint()

        # if percentile == 30:  # mid
        #     prob_mid = datum.true_pred_prob.quantile(85 / 100)
        #     improb_mid = datum.true_pred_prob.quantile(15 / 100)
        #     datum_prob_mid = datum_top[datum_top.true_pred_prob <= prob_mid]
        #     datum_improb_mid = datum_bot[datum_bot.true_pred_prob >= improb_mid]
        # datum_top["word_num"] = datum_top.groupby(datum_top.word).cumcount() + 1
        # datum_bot["word_num"] = datum_bot.groupby(datum_bot.word).cumcount() + 1
        datum_top["word_num"] = (
            datum_top.sort_values(["word", "true_pred_prob"], ascending=False)
            .groupby("word")
            .cumcount()
            + 1
        )
        datum_bot["word_num"] = (
            datum_bot.sort_values(["word", "true_pred_prob"], ascending=True)
            .groupby("word")
            .cumcount()
            + 1
        )
        if "alignednum-improb" in args.datum_mod:  # improb num aligned with prob
            datum = datum_bot.merge(
                datum_top.loc[:, ("word", "word_num")],
                how="inner",
                on=["word", "word_num"],
            )
            print(f"Selected {len(datum.index)} bot pred prob words")
        elif "alignednum-prob" in args.datum_mod:  # improb num aligned with prob
            datum = datum_top.merge(
                datum_bot.loc[:, ("word", "word_num")],
                how="inner",
                on=["word", "word_num"],
            )
            print(f"Selected {len(datum.index)} top pred prob words")
        elif "aligned-improb" in args.datum_mod:  # improb aligned with prob
            datum = datum_bot[datum_bot.word.isin(datum_top.word.unique())]
            print(f"Selected {len(datum.index)} bot pred prob words")
        elif "aligned-prob" in args.datum_mod:  # prob aligned with improb
            datum = datum_top[datum_top.word.isin(datum_bot.word.unique())]
            print(f"Selected {len(datum.index)} top pred prob words")
        # elif "improb-mid" in args.datum_mod:
        #     datum = datum_improb_mid
        #     print(f"Selected {len(datum.index)} mid pred prob words")
        # elif "prob-mid" in args.datum_mod:
        #     datum = datum_prob_mid
        #     print(f"Selected {len(datum.index)} mid pred prob words")
        elif "improb" in args.datum_mod:  # improb
            datum = datum_bot
            print(f"Selected {len(datum.index)} bot pred prob words")
        elif "prob" in args.datum_mod:  # prob
            datum = datum_top
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
    partial = args[args.find(mode) + len(mode) :]

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
    # Average Embeddings per word
    if (  # HACK 1 & 2 turn off, moved earlier for prob-improb
        "glove" not in args.emb_type  # glove emb
        and "glove50" not in args.align_with  # aligned with glove emb
        and "glove" not in args.emb_mod  # replaced with glove emb (already aligned)
    ):
        if "first" in args.emb_mod:
            idx = (
                datum.groupby(["adjusted_onset", "word"], sort=False)[
                    "token_idx"
                ].transform(min)
                == datum["token_idx"]
            )
            datum = datum[idx]
        else:
            datum = ave_emb(datum)  # average embs per word

    datum = get_pos(datum)

    ## Token manipulation
    if "-all" in args.datum_mod:  # all tokens
        pass

    elif "-zeroshot" in args.datum_mod:  # zeroshot tokens
        datum = zeroshot_datum(datum)

    else:  # modify datum based on predictions
        datum = mod_datum_by_preds(args, datum)

    # else:
    #     raise Exception('Invalid Datum Modification')

    # Normalize Embeddings
    if args.normalize:
        datum = normalize_embeddings(args, datum)

    assert len(datum.index) > 0, "Empty Datum"
    return datum


def run_pca(args, df):
    pca_to = args.pca_to
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)
    print(f"PCA from {embs.shape[1]} to {pca_to}")
    pca_output = pca.fit_transform(embs)
    print(f"PCA explained variance: {sum(pca.explained_variance_)}")
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_)}")
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
    base_df = load_datum(args.base_df_path)
    emb_df = load_datum(args.emb_df_path)

    if len(emb_df) != len(base_df):
        df = pd.merge(
            base_df, emb_df, left_index=True, right_index=True
        )  # TODO Needs testing (either bert_utterance or whisper)
    else:
        base_df.reset_index(drop=False, inplace=True)
        df = pd.concat([base_df, emb_df], axis=1)
    print(f"After loading: Datum loads with {len(df)} words")

    df = process_conversations(args, df, stitch)
    # df = ave_emb_preds(df)  # HACK 2
    df = process_embeddings(args, df)
    print(f"After processing: Datum now has {len(df)} words")

    ######################
    # if "concat-emb" in args.emb_mod and "improb" in args.emb_mod:  # concat improb
    #     print(f"Running early pca due to big embedding dimension")
    #     # df = run_pca(args, df)
    #     df = concat_emb_improb(args, df, "concat-emb-improb")

    #######################

    df = filter_datum(args, df)
    print(f"After filtering: Datum now has {len(df)} words")
    df = mod_datum(args, df)  # further filter datum based on datum_mod argument
    if "-pos" in args.datum_mod:
        keep_pos = ["NOUN", "VERB", "ADJ", "ADP", "ADV"]
        df = df[df.part_of_speech.isin(keep_pos)]
    print(f"Datum final length: {len(df)}")

    if "earlypca" in args.datum_mod:  # pca
        print(f"Running early pca due to big embedding dimension")
        df = run_pca(args, df)
    # else:
    #     df.drop(columns="embeddings", inplace=True)
    #     df.to_pickle(f"{args.sid}_gpt2_32.pkl")
    #     breakpoint()

    if "concat-emb" in args.emb_mod and "improb" not in args.emb_mod:  # regular concat
        df = concat_emb(args, df, "concat-emb")

    elif "concat-emb" in args.emb_mod and "improb" in args.emb_mod:  # concat improb
        df = concat_emb_improb(args, df, "concat-emb-improb")

    return df
