import argparse
import csv
import glob
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--output-dir", type=str, default="results/tfs/new-diff")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

files1 = glob.glob(args.formats[0] + "/*.csv")
files2 = glob.glob(args.formats[1] + "/*.csv")
assert len(files1) == len(files2), "Need same number files under data sources"

for file in files1:
    if "summary" in file:
        continue
    filename = os.path.basename(file)  # get filename
    newfilename = args.output_dir + filename

    df = pd.read_csv(file, header=None)
    file2 = args.formats[1] + filename
    df2 = pd.read_csv(file2, header=None)
    df_new = df - df2

    with open(newfilename, "w") as csvfile:
        print(f"calc diff for {filename}")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([df_new.loc[0].tolist()])
