import argparse
import glob
import itertools
import os
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------


def arg_parser():  # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--sid", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument("--fig-size", nargs="+", type=int, default=[18, 6])
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--x-vals-show", nargs="+", type=float, required=True)
    parser.add_argument("--lag-ticks", nargs="+", type=float, default=[])
    parser.add_argument("--lag-tick-labels", nargs="+", type=int, default=[])
    parser.add_argument("--lc-by", type=str, default=None)
    parser.add_argument("--ls-by", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--split-by", type=str, default=None)
    parser.add_argument("--outfile", default="results/figures/tmp.pdf")
    args = parser.parse_args()
    return args


def arg_assert(args):  # some sanity checks
    assert len(args.fig_size) == 2
    assert len(args.formats) == len(
        args.labels
    ), "Need same number of labels as formats"
    assert len(args.lags_show) == len(
        args.x_vals_show
    ), "Need same number of lags values and x values"
    assert all(
        lag in args.lags_plot for lag in args.lags_show
    ), "Lags plot should contain all lags from lags show"
    assert all(
        lag in x_vals_show for lag in args.lag_ticks
    ), "X values show should contain all values from lags ticks"
    assert all(
        lag in lags_show for lag in args.lag_tick_labels
    ), "Lags show should contain all values from lag tick labels"
    assert len(args.lag_ticks) == len(
        args.lag_tick_labels
    ), "Need same number of lag ticks and lag tick labels"

    if args.split:
        assert args.split_by, "Need split by criteria"
        assert args.split == "horizontal" or args.split == "vertical"
        assert args.split_by == "keys" or args.split_by == "labels"


args = arg_parser()
x_vals_show = [x_val / 1000 for x_val in args.x_vals_show]
lags_show = [lag / 1000 for lag in args.lags_show]
arg_assert(args)


# -----------------------------------------------------------------------------
# Functions for Color and Style Maps
# -----------------------------------------------------------------------------


def colorFader(c1, c2, mix):
    """Get color in between two colors (based on linear interpolate)

    Args:
        c1: color 1 in hex format
        c2: color 2 in hex format
        mix: percentage between two colors (0 is c1, 1 is c2)

    Returns:
        a color in hex format
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_cmap_smap(args):
    """Get line color and style map for given label key combinations

    Args:
        args (namespace): commandline arguments

    Returns:
        cmap: dictionary of {line color: (label, key)}
        smap: dictionary of {line style: (label, key)}
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]  # separate colors

    cmap = plt.cm.get_cmap("jet")

    if len(unique_labels) > 10:
        colors = [
            cmap(i / len(unique_labels)) for i in range(0, len(unique_labels))
        ]
    # colors = [colorFader('#97baf7','#000308',i/col_len) for i in range(1,col_len)] # color gradient

    # colors2 = [colorFader('#97baf7','#032661',i/col_len) for i in range(1,col_len)] # color gradient
    # colors1 = [colorFader('#f2b5b1','#6e0801',i/col_len) for i in range(1,col_len)]
    # colors = colors1 + colors2
    styles = ["-", "--", "-.", ":"]
    cmap = {}  # line color map
    smap = {}  # line style map

    if (
        args.lc_by == "labels" and args.ls_by == "keys"
    ):  # line color by labels and line style by keys
        for label, color in zip(unique_labels, colors):
            for key, style in zip(unique_keys, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == "keys" and args.ls_by == "labels"
    ):  # line color by keys and line style by labels
        for key, color in zip(unique_keys, colors):
            for label, style in zip(unique_labels, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "labels"
    ):  # both line color and style by labels
        for label, color, style in zip(unique_labels, colors, styles):
            for key in unique_keys:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "keys"
    ):  # both line color and style by keys
        for key, color, style in zip(unique_keys, colors, styles):
            for label in unique_labels:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    else:
        raise Exception("Invalid input for arguments lc_by or ls_by")
    return (cmap, smap)


unique_labels = list(dict.fromkeys(args.labels))
unique_keys = list(dict.fromkeys(args.keys))
cmap, smap = get_cmap_smap(args)


# -----------------------------------------------------------------------------
# Read Significant Electrode Files
# -----------------------------------------------------------------------------

sigelecs = {}
multiple_sid = False  # only 1 subject
if len(args.sid) > 1:
    multiple_sid = True  # multiple subjects
if len(args.sig_elec_file) == 0:
    pass
elif len(args.sig_elec_file) == len(args.sid) * len(args.keys):
    sid_key_tup = [x for x in itertools.product(args.sid, args.keys)]
    for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
        sig_file = pd.read_csv("data/" + fname)
        if sig_file.subject.nunique() == 1:
            # elecs = sig_file['electrode'].tolist()
            # sigelecs[sid_key] = set(elecs) # need to use this for old 625-676 results
            sig_file["sid_electrode"] = (
                sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
            )
            elecs = sig_file["sid_electrode"].tolist()
            sigelecs[sid_key] = set(elecs)
        else:
            sig_file["sid_electrode"] = (
                sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
            )
            elecs = sig_file["sid_electrode"].tolist()
            sigelecs[sid_key] = set(elecs)
            multiple_sid = True
else:
    raise Exception(
        "Need a significant electrode file for each subject-key combo"
    )


# -----------------------------------------------------------------------------
# Aggregate Data
# -----------------------------------------------------------------------------

print("Aggregating data")
data = []

for fmt, label in zip(args.formats, args.labels):
    load_sid = 0
    for sid in args.sid:
        if str(sid) in fmt:
            load_sid = sid
    assert (
        load_sid != 0
    ), f"Need subject id for format {fmt}"  # check subject id for format is provided
    for key in args.keys:
        fname = fmt % key
        files = glob.glob(fname)
        assert (
            len(files) > 0
        ), f"No results found under {fname}"  # check files exist under format

        for resultfn in files:
            elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
            # Skip electrodes if they're not part of the sig list
            # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
            #     continue
            if len(sigelecs) and elec not in sigelecs[(load_sid, key)]:
                continue
            df = pd.read_csv(resultfn, header=None)
            df.insert(0, "sid", load_sid)
            df.insert(0, "mode", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)

if not len(data):
    print("No data found")
    exit(1)
df = pd.concat(data)
df.set_index(["label", "electrode", "mode", "sid"], inplace=True)

n_lags, n_df = len(args.lags_plot), len(df.columns)
assert (
    n_lags == n_df
), "args.lags_plot length ({n_av}) must be the same size as results ({n_df})"


def add_sid(df, elec_name_dict):
    elec_name = df.index.to_series().str.get(1).tolist()
    sid_name = df.index.to_series().str.get(3).tolist()
    for idx, string in enumerate(elec_name):
        if string.find("_") < 0 or not string[0:3].isdigit():  # no sid in front
            new_string = str(sid_name[idx]) + "_" + string  # add sid
            elec_name_dict[string] = new_string
    return elec_name_dict


elec_name_dict = {}
# new_sid = df.index.to_series().str.get(1).apply(add_sid) # add sid if no sid in front
elec_name_dict = add_sid(df, elec_name_dict)
df = df.rename(index=elec_name_dict)  # rename electrodes to add sid in front

if len(args.lags_show) < len(
    args.lags_plot
):  # if we want to plot part of the lags and not all lags
    print("Trimming Data")
    chosen_lag_idx = [
        idx
        for idx, element in enumerate(args.lags_plot)
        if element in args.lags_show
    ]
    df = df.loc[:, chosen_lag_idx]  # chose from lags to show for the plot
    assert len(x_vals_show) == len(
        df.columns
    ), "args.lags_show length must be the same size as trimmed df column number"


# -----------------------------------------------------------------------------
# Plotting Average and Individual Electrodes
# -----------------------------------------------------------------------------


def sep_sid_elec(string):
    """Separate string into subject id and electrode name

    Args:
        string: string in the format

    Returns:
        tuple in the format (subject id, electrode name)
    """
    sid_index = string.find("_")
    if sid_index > 1:  # if string contains '_'
        if string[:sid_index].isdigit():  # if electrode name starts with sid
            sid_name = string[:sid_index]
            elec_name = string[(sid_index + 1) :]  # remove the sid
    return (sid_name, elec_name)


def get_elecbrain(electrode):
    """Get filepath for small brain plots

    Args:
        electrode: electrode name

    Returns:
        imname: filepath for small brain plot for the given electrode
    """
    sid, electrode = sep_sid_elec(electrode)
    if sid == "7170":
        sid = "717"
    elecdir = f"/projects/HASSON/247/data/elecimg/{sid}/"
    name = electrode.replace("EEG", "").replace("REF", "").replace("\\", "")
    name = name.replace("_", "").replace("GR", "G")
    imname = elecdir + f"thumb_{name}.png"  # + f'{args.sid}_{name}.png'
    return imname


def plot_average(pdf):
    print("Plotting Average")
    fig, ax = plt.subplots(figsize=fig_size)
    # axins = inset_axes(ax, width=3, height=1.5, loc=2, borderpad=4)
    for mode, subdf in df.groupby(["label", "mode"], axis=0):
        vals = subdf.mean(axis=0)
        err = subdf.sem(axis=0)
        label = "-".join(mode)
        ax.fill_between(
            x_vals_show, vals - err, vals + err, alpha=0.2, color=cmap[mode]
        )
        ax.plot(
            x_vals_show,
            vals,
            label=f"{label} ({len(subdf)})",
            color=cmap[mode],
            ls=smap[mode],
        )
        # layer_num = int(mode[0].replace("layer", ""))
        # axins.scatter(layer_num, max(vals), color=cmap[mode])
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.legend(loc="upper right", frameon=False)
    ax.set(xlabel="Lag (s)", ylabel="Correlation (r)", title="Global average")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_split_by_key(pdf, split_dir):
    if split_dir == "horizontal":
        print("Plotting Average split horizontally by keys")
        fig, axes = plt.subplots(1, len(unique_keys), figsize=fig_size)
    else:
        print("Plotting Average split vertically by keys")
        fig, axes = plt.subplots(len(unique_keys), 1, figsize=fig_size)
    for ax, (mode, subdf) in zip(axes, df.groupby("mode", axis=0)):
        for label, subsubdf in subdf.groupby("label", axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            key = (label, mode)
            ax.fill_between(
                x_vals_show, vals - err, vals + err, alpha=0.2, color=cmap[key]
            )
            ax.plot(
                x_vals_show,
                vals,
                label=f"{label} ({len(subsubdf)})",
                color=cmap[key],
                ls=smap[key],
            )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(mode + " global average")
        ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_average_split_by_label(pdf, split_dir):
    if split_dir == "horizontal":
        print("Plotting Average split horizontally by labels")
        fig, axes = plt.subplots(1, len(unique_labels), figsize=fig_size)
    else:
        print("Plotting Average split vertically by labels")
        fig, axes = plt.subplots(len(unique_labels), 1, figsize=fig_size)
    for ax, (label, subdf) in zip(axes, df.groupby("label", axis=0)):
        for mode, subsubdf in subdf.groupby("mode", axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            key = (label, mode)
            ax.fill_between(
                x_vals_show, vals - err, vals + err, alpha=0.2, color=cmap[key]
            )
            ax.plot(
                x_vals_show,
                vals,
                label=f"{mode} ({len(subsubdf)})",
                color=cmap[key],
                ls=smap[key],
            )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(label + " global average")
        ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes(pdf):
    print("Plotting Individual Electrodes")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, ax = plt.subplots(figsize=fig_size)
        # axins = inset_axes(ax, width=3, height=1.5, borderpad=4)
        for (label, _, mode, _), values in subdf.iterrows():
            mode = (label, mode)
            label = "-".join(mode)
            ax.plot(
                x_vals_show,
                values,
                label=label,
                color=cmap[mode],
                ls=smap[mode],
            )
            # layer_num = int(mode[0].replace("layer", ""))
            # axins.scatter(layer_num, max(values), color=cmap[mode])
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
        ax.legend(loc="upper left", frameon=False)
        ax.set(
            xlabel="Lag (s)",
            ylabel="Correlation (r)",
            title=f"{sid} {electrode}",
        )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )
        pdf.savefig(fig)
        plt.close()
    return pdf


def plot_electrodes_split_by_key(pdf, split_dir):
    print("Plotting Individual Electrodes split by keys")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        if split_dir == "horizontal":
            fig, axes = plt.subplots(1, len(unique_keys), figsize=fig_size)
        else:
            fig, axes = plt.subplots(len(unique_keys), 1, figsize=fig_size)
        for ax, (mode, subsubdf) in zip(axes, subdf.groupby("mode")):
            for row, values in subsubdf.iterrows():
                label = row[0]
                key = (label, mode)
                ax.plot(
                    x_vals_show,
                    values,
                    label=label,
                    color=cmap[key],
                    ls=smap[key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper left", frameon=False)
            ax.set_ylim(-0.05, 0.3)  # .35
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {mode}",
            )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )
        pdf.savefig(fig)
        plt.close()
    return pdf


def plot_electrodes_split_by_label(pdf, split_dir):
    print("Plotting Individual Electrodes split by labels")
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        if split_dir == "horizontal":
            fig, axes = plt.subplots(1, len(unique_labels), figsize=fig_size)
        else:
            fig, axes = plt.subplots(len(unique_labels), 1, figsize=fig_size)
        for ax, (label, subsubdf) in zip(axes, subdf.groupby("label")):
            for row, values in subsubdf.iterrows():
                mode = row[2]
                key = (label, mode)
                ax.plot(
                    x_vals_show,
                    values,
                    label=mode,
                    color=cmap[key],
                    ls=smap[key],
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper left", frameon=False)
            ax.set_ylim(vmin - 0.05, vmax + 0.05)  # .35
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {label}",
            )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )
        pdf.savefig(fig)
        plt.close()
    return pdf


pdf = PdfPages(args.outfile)
fig_size = (args.fig_size[0], args.fig_size[1])
vmax, vmin = df.max().max(), df.min().min()
if args.split:
    if args.split_by == "keys":
        pdf = plot_average_split_by_key(pdf, args.split)
        pdf = plot_electrodes_split_by_key(pdf, args.split)
    elif args.split_by == "labels":
        pdf = plot_average_split_by_label(pdf, args.split)
        pdf = plot_electrodes_split_by_label(pdf, args.split)
else:
    pdf = plot_average(pdf)
    pdf = plot_electrodes(pdf)

pdf.close()
