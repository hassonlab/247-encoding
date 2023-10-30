import argparse
import csv
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def main():
    df = pd.read_csv("df.csv")
    df.columns = ["window_index", "nums"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    axes.hist(df["nums"], bins=auto, density=True, cumulative=True)
    plt.savefig("plt.jpeg")
    breakpoint()
    return


if __name__ == "__main__":
    main()
