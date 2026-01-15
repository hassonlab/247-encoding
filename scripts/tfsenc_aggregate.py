import os
import glob

import argparse
import pandas as pd
import numpy as np
import shutil

comp_cor_all = pd.DataFrame()
prod_cor_all = pd.DataFrame()
comp_cor_folds_all = pd.DataFrame()
prod_cor_folds_all = pd.DataFrame()


def parse_arguments():
    """Read arguments from yaml config file

    Returns:
        namespace: all arguments from yaml config file
    """
    # parse yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--label1", nargs="*", type=str, default="[]")
    parser.add_argument("--label2", nargs="*", type=str, default="[]")
    parser.add_argument("--format", nargs="?", type=str, default="")
    args = parser.parse_args()

    return args


def read_dir(folder_dir):
    """Aggregate electrode results from one folder directory

    Returns:
        folder_dir (string): folder directory
    """

    comp_files = glob.glob(f"{folder_dir}/*/*_comp.csv")
    prod_files = glob.glob(f"{folder_dir}/*/*_prod.csv")

    def append_elec_results(elec_files):
        cor = pd.DataFrame()
        cor_folds = pd.DataFrame()

        for elec_file in elec_files:
            elec_info = os.path.basename(elec_file)[:-9].split("_", 1)

            elec_cor = pd.read_csv(elec_file, header=None)
            elec_cor["subject"] = elec_info[0]
            elec_cor["electrode"] = elec_info[1]

            cor = pd.concat((cor, elec_cor.tail(1)))

            num_folds = len(elec_cor) - 1
            elec_cor["fold"] = np.arange(0, num_folds + 1)
            cor_folds = pd.concat((cor_folds, elec_cor[0:num_folds]))
        return cor, cor_folds

    comp_cor = pd.DataFrame()
    comp_cor_folds = pd.DataFrame()
    prod_cor = pd.DataFrame()
    prod_cor_folds = pd.DataFrame()
    if len(comp_files) > 0:
        comp_cor, comp_cor_folds = append_elec_results(comp_files)
    if len(prod_files) > 0:
        prod_cor, prod_cor_folds = append_elec_results(prod_files)

    return comp_cor, comp_cor_folds, prod_cor, prod_cor_folds


def append_dir(folder_dir, remove, label1="", label2=""):
    """Append electrode results from one folder directory
    Also save config and summary files

    Returns:
        folder_dir (str): folder directory
        remove (binary): whether to remove the folder after concatenating
        label1 (str): label 1
        label2 (str): label 2
    """

    comp_cor, comp_cor_folds, prod_cor, prod_cor_folds = read_dir(folder_dir)
    if len(label2) > 0:
        comp_cor["label2"] = label2
        comp_cor_folds["label2"] = label2
        prod_cor["label2"] = label2
        prod_cor_folds["label2"] = label2
    if len(label1) > 0:
        comp_cor["label1"] = label1
        comp_cor_folds["label1"] = label1
        prod_cor["label1"] = label1
        prod_cor_folds["label1"] = label1

    global comp_cor_all, comp_cor_folds_all, prod_cor_all, prod_cor_folds_all
    if len(comp_cor) > 1:
        comp_cor_all = pd.concat((comp_cor_all, comp_cor))
        comp_cor_folds_all = pd.concat((comp_cor_folds_all, comp_cor_folds))
    if len(prod_cor) > 1:
        prod_cor_all = pd.concat((prod_cor_all, prod_cor))
        prod_cor_folds_all = pd.concat((prod_cor_folds_all, prod_cor_folds))

    try:
        configfile = glob.glob(f"{folder_dir}/*/config.yml")[0]
        os.rename(configfile, f"{folder_dir}_config.yml")
    except:
        print("Failed to save config file")

    try:
        summaryfile = glob.glob(f"{folder_dir}/*/summary.csv")[0]
        os.rename(summaryfile, f"{folder_dir}_summary.csv")
    except:
        print("Failed to save summary file")

    if remove:
        print(f"Deleting {folder_dir}")
        shutil.rmtree(folder_dir, ignore_errors=True)

    return


def confirm_prompt(question: str) -> bool:
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ").casefold()
    return reply == "y"


def main():

    # Deleting embeddings after concatenation
    remove = False
    if confirm_prompt("Do you want to delete the original files after aggregating?"):
        remove = True

    # Read command line arguments
    args = parse_arguments()

    # Append results from directories
    if len(args.label1) > 0:
        for label1 in args.label1:
            if len(args.label2) > 0:
                for label2 in args.label2:
                    folder_dir = args.format % (label1, label2)
                    append_dir(folder_dir, remove, label1, label2)
            else:
                folder_dir = args.format % label1
                append_dir(folder_dir, remove, label1)
    else:
        append_dir(args.format, remove)

    # Save results
    if len(comp_cor_all) > 1:
        comp_cor_all.to_csv(f"{args.format}_comp.csv", index=False)
        comp_cor_folds_all.to_csv(f"{args.format}_comp_fold.csv", index=False)
    if len(prod_cor_all) > 1:
        prod_cor_all.to_csv(f"{args.format}_prod.csv", index=False)
        prod_cor_folds_all.to_csv(f"{args.format}_prod_fold.csv", index=False)

    return


if __name__ == "__main__":
    main()
