import os
import pandas as pd
import numpy as np


def main():

    sids = [625, 676, 7170, 798]
    for sid in sids:
        df = pd.read_pickle(f"data/tfs/{sid}/pickles/{sid}_full_labels.pkl")
        df = pd.DataFrame(df["labels"])

        df["utt_onset"] = df.groupby([df.conversation_id, df.sentence_idx])[
            "adjusted_onset"
        ].transform("min")
        df["utt_offset"] = df.groupby([df.conversation_id, df.sentence_idx])[
            "adjusted_offset"
        ].transform("max")
        df["word_idx"] = (
            df.groupby([df.conversation_id, df.sentence_idx]).cumcount() + 1
        )
        df["word_idx_r"] = (
            df.groupby([df.conversation_id, df.sentence_idx]).cumcount(ascending=False)
            + 1
        )
        df["word_len"] = df.adjusted_offset - df.adjusted_onset
        df["gap_len"] = df.adjusted_onset - df.adjusted_offset.shift(1)

        # No gap length for the first word of the conversation
        first_word_idx = df.conversation_id.ne(df.conversation_id.shift(1))
        df.loc[first_word_idx, "gap_len"] = np.nan
        df.to_pickle(f"results/tfs/saved_pickles/{sid}_length_labels.pkl")

    return


if __name__ == "__main__":
    main()
