import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle


sub = 676

keys = ["prod", "comp"]

corr = "ave"

saving_elec_file = False  # DO NOT TURN THIS ON OR YOU DIE

formats = [
    '/scratch/gpfs/kw1166/results/tfs/kw-tfs-full-en-onset-625-whisper-tiny.en-l4-wn2-6'
    'results/tfs/kw-tfs-full-' + sub + '-gpt2-xl-triple/kw-200ms-all-' + sub + '/',
    'results/tfs/kw-tfs-full-' + sub + '-blenderbot-small-triple/kw-200ms-all-' + sub + '/'
    ]
# formats = [
#     f"results/tfs/stock/kw-tfs-full-{sub}-erp-lag2k-25-all-test/*/",
# ]

comp_sig_file = f"data/tfs-sig-file-{sub}-sig-1.0-comp.csv"
prod_sig_file = f"data/tfs-sig-file-{sub}-sig-1.0-prod.csv"

coordinatefilename = f"data/brainplot/{sub}_{corr}.txt"
elecfilename = f"data/brainplot/{sub}_elecs.csv"

#################################################################################################
###################################### Compare two sources ######################################
#################################################################################################

data = pd.read_csv(coordinatefilename, sep=" ", header=None)
data = data.set_index(0)
data = data.loc[:, 1:4]
print(f"\nFor subject {sub}:\ntxt has {len(data.index)} electrodes")

if saving_elec_file:
    print("YOU DIED")
    breakpoint()
    files = glob.glob(formats[0] + "*_prod.csv")
    files = [os.path.basename(file) for file in files]
    print(f"encoding has {len(files)} electrodes")

    files = pd.DataFrame(data=files)
    files["elec"] = ""
    files["elec2"] = ""
    for row, values in files.iterrows():
        elec = os.path.basename(values[0]).replace(".csv", "")[:-5]
        files.loc[row, "elec"] = elec
        # elec = elec[:-3] # just for 625 and 676
        elec = elec.replace("EEG", "").replace("GR_", "G").replace("_", "")
        files.loc[row, "elec2"] = elec
    files = files.set_index(0)
    files = files.sort_values(by="elec2")
    # files.to_csv(elecfilename, index=False)

    breakpoint()
    # go to the file and fix everything

    elecs = pd.read_csv(elecfilename)
    set1 = set(data.index)
    set2 = set(files.elec2)
    set3 = set(elecs.dropna(subset="elec2").elec2)
    set4 = set(elecs.dropna(subset="elec").elec)

    breakpoint()
    print(f"txt and encoding share {len(set1.intersection(set2))} electrodes\n")
    print(f"encoding does not have these electrodes: {sorted(set1-set2)}\n")
    print(f"txt does not have these electrodes: {sorted(set2-set1)}\n")


#############################################################################################
###################################### Getting Results ######################################
#############################################################################################

elecs = pd.read_csv(elecfilename)
elecs = elecs.dropna()
elecs = elecs.rename(columns={"elec2": 0})
elecs.set_index(0, inplace=True)

df = pd.merge(data, elecs, left_index=True, right_index=True)

df["comp"] = 100
df["prod"] = 100
df["comp_sig"] = 0
df["prod_sig"] = 0

if sub != 798:
    comp_sig_elecs = pd.read_csv(comp_sig_file)["electrode"].tolist()
    prod_sig_elecs = pd.read_csv(prod_sig_file)["electrode"].tolist()
else:
    comp_sig_elecs = []
    prod_sig_elecs = []

# ################# GET area for a curve #################
# area = pd.read_csv(
#     f"results/brainplot/area_-500_-100_n-1_gloveselect/{sub}_area_gptn-1_glove.csv"
# )
# area = area.drop(["label"], errors="ignore")
# area["elec_name"] = area.electrode.str[len(str(sub)) + 1 :]
# area.set_index("elec_name", inplace=True)

# area_prod = area.loc[area["mode"] == "prod", "area_diff"]
# area_comp = area.loc[area["mode"] == "comp", "area_diff"]

# for row, values in df.iterrows():
#     if values["elec"] in comp_sig_elecs:
#         df.loc[row, "comp_sig"] = 1
#     if values["elec"] in prod_sig_elecs:
#         df.loc[row, "prod_sig"] = 1
#     try:
#         df.loc[row, "comp"] = area_comp[values["elec"]]
#     except:
#         print(row, values["elec"])
#     try:
#         df.loc[row, "prod"] = area_prod[values["elec"]]
#     except:
#         print(row, values["elec"])


# def save_area_results(sub, df, outname, mode, sig=False):
#     df = df.loc[df[mode] < 100, :]
#     sig_string = ""
#     if sig:
#         sig_string = "_sig"
#         df = df.loc[df[mode + sig_string] == 1, :]  # choose only sig electrodes
#     outname = f"{outname}{sub}_{mode}{sig_string}.txt"

#     with open(outname, "w") as outfile:
#         df = df.loc[:, [1, 2, 3, 4, mode]]
#         df.to_string(outfile)


# outname = "results/brainplot/area_-500_-100_n-1_gloveselect/"
# if corr == "ind":
#     outname = outname + "ind/"

# for key in keys:
#     save_area_results(sub, df, outname, key)
#     # save_area_results(sub, df, outname, key, True)


################# GET ERP correlation between prod/comp #################
def get_erp_corr(compfile, prodfile, path):

    filename = os.path.join(path, compfile)
    comp_data = pd.read_csv(filename, header=None)
    filename = os.path.join(path, prodfile)
    prod_data = pd.read_csv(filename, header=None)
    corr_erp, _ = pearsonr(comp_data.loc[0, :], prod_data.loc[0, :])

    return corr_erp


# df["erp"] = -1

# for format in formats:
#     for row, values in df.iterrows():
#         if row in prod_sig_elecs or row in comp_sig_elecs:
#             prod_name = values[0]
#             comp_name = values[0].replace("prod", "comp")
#             df.loc[row, "erp"] = get_erp_corr(comp_name, prod_name, format)

# output_filename = "results/cor_tfs/" + sub + "_" + corr + "_erp_sig" + ".txt"
# with open(output_filename, "w") as outfile:
#     df = df.loc[:, [1, 2, 3, 4, "erp"]]
#     df.to_string(outfile)
# breakpoint()
#####################################################################################


embs = ["glove", "gpt", "bbot"]

emb_key = [emb + "_" + key for emb in embs for key in keys]
for col in emb_key:
    df[col] = -1


def get_max(filename, path):
    filename = os.path.join(path, filename)
    elec_data = pd.read_csv(filename, header=None)
    return max(elec_data.loc[0])


for format in formats:
    if "glove50" in format:
        col_name = "glove"
    elif "gpt2-xl" in format:
        col_name = "gpt"
    elif "blenderbot-small" in format:
        col_name = "bbot"
    print(f"getting results for {col_name} embedding")
    for row, values in df.iterrows():
        col_name1 = col_name + "_prod"
        col_name2 = col_name + "_comp"
        prod_name = values[0]
        comp_name = values[0].replace("prod", "comp")
        if row in prod_sig_elecs:
            df.loc[row, col_name1] = get_max(prod_name, format)
        if row in comp_sig_elecs:
            df.loc[row, col_name2] = get_max(comp_name, format)


for col in emb_key:
    output_filename = "results/cor_tfs/" + sub + "_" + corr + "_" + col + ".txt"
    df_output = df.loc[:, [1, 2, 3, 4, col]]
    with open(output_filename, "w") as outfile:
        df_output.to_string(outfile)
