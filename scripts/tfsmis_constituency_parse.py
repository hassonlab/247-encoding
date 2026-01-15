import pandas as pd
import numpy as np
import string
import stanza
import argparse
from nltk.tree import Tree

PUNCS = '!"#$%&()*+,-—./:;<=>?@[\\]^_`{|}~'


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", nargs="?", type=int, default=None)
    parser.add_argument("--chunk", nargs="?", type=int, default=None)
    args = parser.parse_args()
    return args


def fix_sent_df(sid):
    sent_df = pd.read_csv(f"{sid}_sent_df.csv")
    sent_df = sent_df.loc[
        :, ("word", "adjusted_onset", "sentence", "new_sentence_idx", "new_sentence")
    ]
    sent_df["correct_sent"] = True
    sent_df.drop_duplicates(subset=["word", "adjusted_onset"], inplace=True)
    for row_idx, row in enumerate(sent_df.itertuples()):
        if not pd.isna(row[5]):
            continue
        if row[3] == sent_df.iloc[row_idx - 1, 2] and not pd.isna(
            sent_df.iloc[row_idx - 1, 4]
        ):  # if sent equals to prev and prev not nan
            sent_df.iloc[row_idx, 4] = sent_df.iloc[row_idx - 1, 4]  # new sent
            sent_df.iloc[row_idx, 3] = sent_df.iloc[row_idx - 1, 3]  # new idx
        elif row[3] == sent_df.iloc[row_idx + 1, 2] and not pd.isna(
            sent_df.iloc[row_idx + 1, 4]
        ):  # if sent equals to next and next not nan
            sent_df.iloc[row_idx, 4] = sent_df.iloc[row_idx + 1, 4]  # new sent
            sent_df.iloc[row_idx, 3] = sent_df.iloc[row_idx + 1, 3]  # new idx
        else:
            sent_df.iloc[row_idx, 4] = sent_df.iloc[row_idx, 2]  # original sent
            sent_df.iloc[row_idx, 5] = False  # mark incorrect
    sent_df.to_csv(f"{sid}_sent_df_cleaned.csv", index=False)
    return


def parse_sentences(nlp, text, sent_idx):
    doc = nlp(text)
    results = pd.DataFrame()
    for sent in doc.sentences:  # loop over sentences
        sent_cons = sent.constituency.children[0]
        nltk_tree = Tree.fromstring(str(sent_cons))
        label_leaves(nltk_tree)

        words = nltk_tree.leaves()
        try:
            bot_labels = [
                nltk_tree[leaf[:-1]].label()
                for leaf in nltk_tree.treepositions("leaves")
            ]
            top_labels = [
                nltk_tree[leaf[0]].label() for leaf in nltk_tree.treepositions("leaves")
            ]
        except:
            assert nltk_tree.height() <= 2 and len(nltk_tree.leaves()) == 1
            bot_labels = [nltk_tree.label()]
            top_labels = [nltk_tree.label()]
            nltk_tree.pretty_print()

        result = pd.DataFrame(
            {"word": words, "bot_label": bot_labels, "top_label": top_labels}
        )
        result["phrase_start"] = result["word"].apply(lambda x: "*FIRST" in x)
        result["phrase_end"] = result["word"].apply(lambda x: "*LAST" in x)
        result["word"] = result["word"].apply(
            lambda x: x.replace("*FIRST*", "")
            .replace("*LAST*", "")
            .replace("*MID*", "")
        )
        result["label_idx"] = result.index.tolist()
        result["label_idx_r"] = result.index[::-1].tolist()
        results = pd.concat([results, result])
    return pd.Series(
        {
            "word": results["word"].tolist(),
            "bot_label": results["bot_label"].tolist(),
            "top_label": results["top_label"].tolist(),
            "phrase_start": results["phrase_start"].tolist(),
            "phrase_end": results["phrase_end"].tolist(),
            "label_idx": results["label_idx"].tolist(),
            "label_idx_r": results["label_idx_r"].tolist(),
            "sentence": text,
            "sentence_idx": sent_idx,
        }
    )


def parse_words(nlp, text):

    doc = nlp(text)
    tokens = [word.text for word in doc.sentences[0].words]
    return tokens


def merge_labels(args):

    sent_df = pd.read_csv(f"{args.sid}_sent_df_tokenized.csv")
    sent_df["token_l"] = sent_df["token"].apply(lambda x: x.lower())
    sent_df["token_l"] = sent_df["token_l"].fillna("")

    def cum_token_counts(words):
        counts = {}
        reps = []
        # print(words)
        for word in words:
            if word not in counts.keys():
                counts[word] = 0
            else:
                counts[word] += 1
            reps.append(counts[word])
        return reps

    # Get token count within sentence
    sent_df["token_cumcount"] = sent_df.groupby("new_sentence_idx_all")[
        "token_l"
    ].transform(lambda s: pd.Series(cum_token_counts(s), index=s.index))

    labels = pd.read_csv(f"{args.sid}_constituency_labels.csv")
    labels.rename(
        columns={
            "word": "token",
            "sentence": "new_sentence",
            "sentence_idx": "new_sentence_idx_all",
        },
        inplace=True,
    )
    labels["token_l"] = labels["token"].apply(lambda x: x.lower())

    # Get token count within sentence
    labels["token_cumcount"] = labels.groupby("new_sentence_idx_all")[
        "token_l"
    ].transform(lambda s: pd.Series(cum_token_counts(s), index=s.index))

    # Strip punctuation for matching
    # labels["token_stripped"] = labels["token_l"].apply(
    #     lambda x: x.translate(str.maketrans("", "", PUNCS))
    # )
    # # labels["token_stripped"] = labels["token_l"].apply(
    # #     lambda x: x.translate(str.maketrans("", "", string.punctuation))
    # # )
    # labels = labels[labels["token_stripped"] != ""]

    print(len(sent_df), len(labels))
    labels.drop(columns=["token", "new_sentence"], inplace=True)
    sent_df = sent_df.merge(
        labels,
        on=["token_l", "new_sentence_idx_all", "token_cumcount"],
        how="left",
    )
    sent_df.to_csv(f"{args.sid}_sent_df_constituency_labels.csv", index=False)
    breakpoint()


def label_leaves(tree):
    """Modify the tree in place to label first and last leaves of NP / VP subtrees."""
    for st in tree.subtrees(lambda t: t.label() == "NP" or t.label() == "VP"):
        leaves = st.leaves()
        if leaves:
            for i, leaf in enumerate(leaves):
                leaf_idx = st.leaf_treeposition(i)
                if i == 0 and "*FIRST" not in st[leaf_idx]:  # first leaf
                    st[leaf_idx] = f"*FIRST*{st[leaf_idx]}"
                if i == len(leaves) - 1 and "*LAST" not in st[leaf_idx]:  # last leaf
                    st[leaf_idx] = f"*LAST*{st[leaf_idx]}"
                if i != 0 and i != len(leaves) - 1:  # middle leaves
                    st[leaf_idx] = f"*MID*{st[leaf_idx]}"
    return tree


def main():

    args = parse_arguments()
    merge_labels(args)

    datum = pd.read_pickle(f"data/tfs/{args.sid}/pickles/{args.sid}_full_labels.pkl")
    datum = pd.DataFrame(datum["labels"])

    # fix_sent_df(args.sid)
    sent_df = pd.read_csv(f"{args.sid}_sent_df_cleaned.csv")
    cols = ["new_sentence", "new_sentence_idx"]
    change_mask = sent_df[cols].ne(sent_df[cols].shift()).any(axis=1)
    sent_df["new_sentence_idx_all"] = change_mask.astype(int).cumsum()
    print(sent_df.isna().sum())

    ####### Tokenizer ######
    nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt", download_method=None)
    sent_df["token"] = sent_df["word"].apply(lambda w: parse_words(nlp, w))
    sent_df = sent_df.explode("token", ignore_index=True)
    sent_df.to_csv(f"{args.sid}_sent_df_tokenized.csv", index=False)
    breakpoint()

    ####### Constituency Parser ######
    sent_df_unique = sent_df.drop_duplicates(
        subset=["new_sentence", "new_sentence_idx_all"]
    ).copy()
    # sent_df_unique = sent_df_unique.iloc[0:10, :]
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,mwt,pos,constituency", download_method=None
    )
    labels = sent_df_unique.apply(
        lambda row: parse_sentences(
            nlp, row["new_sentence"], row["new_sentence_idx_all"]
        ),
        axis=1,
        result_type="expand",
    )
    labels = labels.explode(
        [
            "word",
            "bot_label",
            "top_label",
            "phrase_start",
            "phrase_end",
            "label_idx",
            "label_idx_r",
        ],
        ignore_index=True,
    )
    # labels = labels.loc[labels.word.isin(sent_df["token"])]
    labels.to_csv(f"{args.sid}_constituency_labels.csv", index=False)

    return


if __name__ == "__main__":
    main()
