from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

def get_pn(perm_pkz):
    return int(perm_pkz.parts[-1].split("_")[0].split("-")[1])


def get_crt(perm_pkz):
    pn = int(perm_pkz.parts[-1].split("_")[0].split("-")[1])
    task = perm_pkz.parts[-1].split("_")[2]
    contrast = perm_pkz.parts[-1].split("_")[3]
    run = perm_pkz.parts[-1].split("_")[4]
    return pn, task, contrast, run


def get_varex(df):
    df["agh_varex"] = (df.agh_r2 - df.int_r2) * 100
    df["mfg_varex"] = (df.aghs_r2 - df.agh_r2) * 100
    df["model_varex"] = (df.aghss_r2 - df.aghs_r2) * 100
    df["site_varex"] = (df.aghsss_r2 - df.aghss_r2) * 100
    return df


def make_bar_list(
    factors,
    to_plot,
    to_plot_sig,
    group_offset=1.5,
    left_start=0,
    cols=2,
    sig_palette=sns.color_palette("muted"),
    palette=sns.color_palette("pastel"),
    error=None,
    group_var=None,
    sort_var=None,
    sig_exclude=[],
    stack=True,
):
    """Make input for bar chart that will give two columns of stacked bars"""
    fplist = []
    left = np.zeros(len(to_plot)) + left_start
    height = 0.8
    if not stack:
        height /= len(factors)
    for i, vf in enumerate(factors):
        fdict = {
            "y": [],
            "width": [],
            "left": left.copy(),
            "color": [],
            "label": [],
            "tick_label": [],
            "height": height,
        }
        if error is not None:
            fdict["xerr"] = []
        if group_var is not None:
            fdict[group_var] = []
        yoffset = 0

        if group_var is not None:
            for x, df in to_plot.groupby(group_var):
                if not stack:
                    fdict["y"].extend(list(np.arange(len(df)) + yoffset + height * i))
                else:
                    fdict["y"].extend(list(np.arange(len(df)) + yoffset))
                fdict["width"].extend(list(df[vf].values))
                fdict["tick_label"].extend(list(df.label.values))
                yoffset += len(df) + group_offset
                if group_var is not None:
                    fdict[group_var].extend(list(df[group_var].values))
                labels = np.array([vf] * len(df))
                colors = np.array([palette[i]] * len(df))

                if vf not in sig_exclude:
                    sig_mask = df[vf].values >= to_plot_sig[vf]
                    labels[sig_mask] = "Sig. " + vf
                    colors[sig_mask] = sig_palette[i]
                fdict["color"].extend(list(colors))
                fdict["label"].extend(list(labels))
                if error is not None:
                    fdict["xerr"].extend(list(df[error[i]].values))
        else:
            df = to_plot
            if sort_var is not None:
                df = df.sort_values(sort_var)
            if not stack:
                fdict["y"].extend(list(np.arange(len(df)) + yoffset + height * i))
            else:
                fdict["y"].extend(list(np.arange(len(df)) + yoffset))
            fdict["width"].extend(list(df[vf].values))
            fdict["tick_label"].extend(list(df.label.values))
            if error is not None:
                fdict["xerr"].extend(list(df[error[i]].values))
            yoffset += len(df) + group_offset
            if group_var is not None:
                fdict[group_var].extend(list(df[group_var].values))
            labels = np.array([vf] * len(df))
            colors = np.array([palette[i]] * len(df))
            if vf not in sig_exclude:
                sig_mask = df[vf].values >= to_plot_sig[vf]
                labels[sig_mask] = "Sig. " + vf
                colors[sig_mask] = sig_palette[i]
            fdict["color"].extend(list(colors))
            fdict["label"].extend(list(labels))

        fdict["width"] = np.array(fdict["width"])
        fdict["width"][fdict["width"] < 0] = 0

        fdf = pd.DataFrame(fdict)
        fdf["col"] = (
            cols - 1 - pd.cut(fdf["y"], cols, labels=np.arange(cols)).astype(int)
        )
        if group_var is not None:
            fdf = fdf.merge(
                pd.DataFrame(fdf.groupby(group_var).col.max()).reset_index(),
                on=group_var,
                suffixes=["_bad", ""],
            ).drop("col_bad", axis=1)
        fplist.append(fdf)
        yoffset += group_offset
        if stack:
            left += np.array(fdict["width"])
    return fplist


def off_diag(a):
    return ~np.eye(a.shape[0], dtype=bool)


def make_signed_ps(upper, lower):
    signed_ps = upper.copy()
    signed_ps[upper > lower] = lower[upper > lower] * -1
    return signed_ps


# TODO: is this the correct version of the function?
# def get_cfn_sig(cfns, perm_cfns):
#     # add true data to permuted data and generate null distributions
#     diag_null_upper = np.array(
#         [list(np.diag(cm)) for cm in perm_cfns.mean(1)] + [list(np.diag(cfns.mean(0)))]
#     ).max(1)
#     diag_null_lower = np.array(
#         [list(np.diag(cm)) for cm in perm_cfns.mean(1)] + [list(np.diag(cfns.mean(0)))]
#     ).min(1)

#     od_null_upper = np.array(
#         [list(cm[off_diag(cm)]) for cm in perm_cfns.mean(1)]
#         + [list(cfns.mean(0)[off_diag(cfns.mean(0))])]
#     ).max(1)
#     od_null_lower = np.array(
#         [list(cm[off_diag(cm)]) for cm in perm_cfns.mean(1)]
#         + [list(cfns.mean(0)[off_diag(cfns.mean(0))])]
#     ).min(1)

#     # Calculate p values, correcting for multiple comparisons
#     diag_upper_ps = np.array(
#         [val < diag_null_upper for val in np.diag(cfns.mean(0))]
#     ).mean(1)
#     diag_lower_ps = np.array(
#         [val > diag_null_lower for val in np.diag(cfns.mean(0))]
#     ).mean(1)

#     off_diag_upper_ps = np.array(
#         [val < od_null_upper for val in cfns.mean(0)[off_diag(cfns.mean(0))]]
#     ).mean(1)
#     off_diag_lower_ps = np.array(
#         [val > od_null_lower for val in cfns.mean(0)[off_diag(cfns.mean(0))]]
#     ).mean(1)

#     # Combine upper and lower thresholds
#     diag_signed_ps = make_signed_ps(diag_upper_ps, diag_lower_ps)
#     off_diag_signed_ps = make_signed_ps(off_diag_upper_ps, off_diag_lower_ps)

#     # load diagonal and off diagnal values back into shape
#     cfn_signed_ps = np.ones(cfns.mean(0).shape)
#     cfn_signed_ps[np.diag_indices_from(cfn_signed_ps)] = diag_signed_ps
#     cfn_signed_ps[off_diag(cfn_signed_ps)] = off_diag_signed_ps

#     return cfn_signed_ps


def get_cfn_sig(cfns, perm_cfns):
    null_upper_tail = perm_cfns.mean(1).max(-1).max(-1)
    null_lower_tail = perm_cfns.mean(1).min(-1).min(-1)

    upper_ps = np.array(
        [val <= null_upper_tail for val in cfns.mean(0).flatten()]
    ).mean(1)
    lower_ps = np.array(
        [val >= null_lower_tail for val in cfns.mean(0).flatten()]
    ).mean(1)

    cfn_signed_ps = make_signed_ps(upper_ps, lower_ps).reshape(cfns[0].shape)
    return cfn_signed_ps


def get_cfns(df, name):
    cfns = df.query("pn == 0 & name == @name").cfn.values
    cfns = np.array([c for c in cfns])
    perm_cfns = []
    for pn in df.pn.unique():
        tmp = df.query("pn == @pn & name == @name").cfn.values
        perm_cfns.append(np.array([c for c in tmp]))

    perm_cfns = np.array(perm_cfns)
    return cfns, perm_cfns


# TODO: there are lots of missing variables here. not sure where they are coming from
# def get_collapsed_perms(cfns, perm_cfns):
#     model_cfns = np.array(
#         [collapse_group(cfn, model_lut, model_order, normalize=False) for cfn in cfns]
#     )
#     # model_perm_cfns = np.array([[collapse_group(pcfn, model_lut, model_order, normalize=False) for pcfn in pc] for pc in perm_cfns])
#     mfg_cfns = np.array(
#         [collapse_group(cfn, mfg_lut, mfg_order, normalize=False) for cfn in cfns]
#     )
#     # mfg_perm_cfns = np.array([[collapse_group(pcfn, mfg_lut, mfg_order, normalize=False) for pcfn in pc] for pc in perm_cfns])

#     model_cfns_norm = np.array(
#         [collapse_group(cfn, model_lut, model_order, normalize=True) for cfn in cfns]
#     )
#     model_perm_cfns_norm = np.array(
#         [
#             [
#                 collapse_group(pcfn, model_lut, model_order, normalize=True)
#                 for pcfn in pc
#             ]
#             for pc in perm_cfns
#         ]
#     )
#     mfg_cfns_norm = np.array(
#         [collapse_group(cfn, mfg_lut, mfg_order, normalize=True) for cfn in cfns]
#     )
#     mfg_perm_cfns_norm = np.array(
#         [
#             [collapse_group(pcfn, mfg_lut, mfg_order, normalize=True) for pcfn in pc]
#             for pc in perm_cfns
#         ]
#     )

#     model_cfn_signed_ps_norm = get_cfn_sig(model_cfns_norm, model_perm_cfns_norm)
#     mfg_cfn_signed_ps_norm = get_cfn_sig(mfg_cfns_norm, mfg_perm_cfns_norm)

#     return model_cfns, mfg_cfns, model_cfn_signed_ps_norm, mfg_cfn_signed_ps_norm


## TODO: is this the correct version of the funcion?
# def plot_confusion_matrix(
#     cm,
#     classes,
#     normalize=False,
#     title="Confusion matrix",
#     cmap=plt.cm.Blues,
#     signed_ps=None,
#     sig_thresh=0.025,
# ):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#     """
#     if normalize:
#         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print("Confusion matrix, without normalization")

#     print(cm)

#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = ".2f" if normalize else ".1f"
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         weight = "normal"
#         if signed_ps is not None:
#             if np.abs(signed_ps[i, j]) < sig_thresh:
#                 weight = "bold"
#         if normalize == False:
#             s = format(cm[i, j], fmt)
#         else:
#             s = format(cm[i, j], fmt)
#         plt.text(
#             j,
#             i,
#             s,
#             horizontalalignment="center",
#             va="center",
#             color="white" if cm[i, j] > thresh else "black",
#             fontweight=weight,
#             size="x-small",
#         )

#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    numbers=False,
    symbols=False,
    cmap=plt.cm.Blues,
    signed_ps=None,
    sig_thresh=0.025,
    ax=None,
    fig=None,
    colorbar=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    if ax is None:
        fig, ax = plt.subplots(1)
    if normalize:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    if colorbar:
        fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, size="x-small")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, size="x-small")

    fmt = ".2f" if normalize else ".1f"
    thresh = 1 / 2.0
    if numbers:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            weight = "normal"
            if signed_ps is not None:
                if np.abs(signed_ps[i, j]) < sig_thresh:
                    weight = "bold"
            if normalize == False:
                s = format(cm[i, j], fmt)
            else:
                s = format(cm[i, j], fmt)
            ax.text(
                j,
                i,
                s,
                horizontalalignment="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight=weight,
                size="xx-small",
            )

    elif symbols:
        if signed_ps is not None:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if np.abs(signed_ps[i, j]) < sig_thresh:
                    if np.sign(1 / signed_ps[i, j]) > 0:
                        s = "+"
                    elif np.sign(1 / signed_ps[i, j]) < 0:
                        s = "-"
                    ax.text(
                        j,
                        i,
                        s,
                        horizontalalignment="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight="bold",
                    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return fig, ax, im


def collapse_group(cfn, group_lut, group_order, normalize=False):

    cfndf = pd.DataFrame(data=cfn)
    group_vals = [group_lut[ii] for ii in cfndf.index.values]
    collapsed = (
        cfndf.assign(group=group_vals)
        .groupby("group")
        .sum()
        .T.assign(group=group_vals)
        .groupby("group")
        .sum()
        .T.loc[group_order, group_order]
    ).values
    if normalize:
        collapsed = collapsed.astype("float") / collapsed.sum(axis=1)[:, np.newaxis]
    return collapsed
