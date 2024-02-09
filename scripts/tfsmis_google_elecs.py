import pandas as pd
import numpy as np
import os


def main():

    SID = 798
    model = "glove"

    mode = "prod"
    mode = "comp"

    csv_dir = "data"
    filename = os.path.join(csv_dir, f"tfs-sig-file-{model}-{SID}-{mode}.csv")

    elec_list = ['G101', 'G7', 'G77', 'G17', 'G8', 'G15', 'G78', 'G88', 'G92', 'G95', 'G16', 'G12', 'G18', 'G93', 'G22', 'G90', 'G19', 'G86', 'G110', 'G26', 'AF6', 'G63', 'G64', 'G56', 'G84', 'G79', 'AIT3', 'G83', 'O6', 'G27', 'AIT2', 'G34', 'DPI3', 'G69', 'G6', 'G2', 'DPI2', 'O5', 'G82', 'G66']

    df = pd.DataFrame({"subject": SID, "electrode": elec_list})
    df.to_csv(filename, index=False)

    return


if __name__ == "__main__":
    main()
