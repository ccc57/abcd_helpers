from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ModPermRes = namedtuple("ModPermRes",['pn','fn','name', 'clf', 'ho_score','cfn'])
# VarPermRes = namedtuple("VarPermRes", ["pn", "metric", "int_r2", "agh_r2", "aghs_r2", "aghss_r2", "aghsss_r2"])


def get_pn(perm_pkz):
    return int(perm_pkz.parts[-1].split("_")[0].split("-")[1])


def get_crt(perm_pkz):
    pn = int(perm_pkz.parts[-1].split("_")[0].split("-")[1])
    task = perm_pkz.parts[-1].split("_")[2]
    contrast = perm_pkz.parts[-1].split("_")[3]
    # run = perm_pkz.parts[-1].split("_")[4]
    return pn, task, contrast  # , run


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
    left = np.zeros(len(to_plot))
    try:
        if len(left_start) != cols:
            raise ValueError(
                "If left_start is an iterable, it must be the same length as the number of columns."
            )
        else:
            left_iter = True
    except TypeError:
        left_iter = False
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
                labels = np.array([vf] * len(df)).astype(np.object)
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
            labels = np.array([vf] * len(df)).astype(np.object)
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
        # set lefts
        if left_iter:
            for ci, ll in enumerate(left_start):
                fdf.loc[fdf.col == ci, "left"] += ll
        else:
            fdf["left"] += left_start
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


def plot_tail(tail, cdf):
    y = np.arange(len(tail)) / float(len(tail))
    plt.plot(tail, y)
    plt.plot(
        np.arange(tail.min(), tail.max(), 0.001),
        cdf(np.arange(tail.min(), tail.max(), 0.001)),
    )
    plt.show()


def _run_gpd_p(x, x0=0, side="upper", nx=260, fit_alpha=0.05, plot=False):
    """Fit tail with generalized pareto distribution to get p-value of x0.
    Based on Knijnenburg et al, 2009 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2687965/)
    Here we use a komogorov-smirnof test for equality of distributions 
    instead of Anderson-Darling as used in their paper.
    Parameters
    ==========
    x: array
        Array of values containing the bootstrap or permutation distribution
    x0: float
        The value the distribution is being tested against
    side: string
        Specify the tail of the distribution to be tested must be one of ["upper", "lower"]
    nx: int
        Starting value for the number of excedences to begin counting down from
        while attempting to fit the GPD
    fit_alpha: float
        Alpha used to reject the null hypothesis that the tail of the data
        comes from the fitted GPD.
    Returns
    =======
    p: float
        fitted p-value
    """
    x = np.sort(x)
    fit_p = 0
    n = len(x)
    if nx > len(x):
        nx = len(x)
    if side == "upper":
        epc = np.count_nonzero(x >= x0)
    elif side == "lower":
        epc = np.count_nonzero(x <= x0)
    else:
        raise ValueError(f'side must be one of ["upper", "lower"], you provided {side}')
    if epc >= 10:
        # TODO: binomial estimate of this
        return (epc + 1) / (n + 1)
    while (fit_p < fit_alpha) & (nx > 10):
        nx -= 10
        if side == "upper":
            t = np.mean([x[-1 * nx], x[-1 * nx - 1]])
            tail = x[-1 * nx :] - t
        else:
            t = np.mean([x[nx], x[nx + 1]])
            tail = np.sort((x[:nx]) - t)
        fit_params = stats.genpareto.fit(tail)
        fitted_gpd = stats.genpareto(*fit_params)
        k = fitted_gpd.args[2]
        fit_stat, fit_p = stats.kstest(tail, fitted_gpd.cdf)
    if fit_p < fit_alpha:
        print(
            "Could not fit GPD to tail of distribution, returning empirical cdf based p.",
            flush=True,
        )
        return (epc + 1) / (n + 1)
        # raise Exception("Could not fit GPD to tail of distribution")

    if plot:
        plot_tail(tail, fitted_gpd.cdf)

    if side == "upper":
        p = nx / n * (1 - fitted_gpd.cdf(x0 - t))
        # If p == 0 and K > 0 then we're in a domain where
        # GPD is finite and unsuitable for extrapolation
        # In these cases, return the pvalue for the extreme of x,
        # which will be conservative
        if (p == 0) & (k > 0):
            p = nx / n * (1 - fitted_gpd.cdf(x.max() - t))
            if p == 0:
                return (epc + 1) / (n + 1)
                # raise Exception("p = 0")
        elif (p == 0) & (k <= 0):
            raise Exception("p=0 and k is not > 0")
    else:
        p = nx / n * (fitted_gpd.cdf(x0 - t))
        if (p == 0) & (k > 0):
            p = nx / n * (fitted_gpd.cdf(x.min() - t))
            if p == 0:
                return (epc + 1) / (n + 1)
                # raise Exception("p = 0")
        elif (p == 0) & (k <= 0):
            raise Exception("p=0 and k is not > 0")

    # return nx, t, fitted_gpd, p
    return p


def test_run_gpd_p():
    np.random.seed(seed=1)
    assert _run_gpd_p(np.random.standard_normal(1000) + 3, side="lower") < 0.05
    assert _run_gpd_p(np.random.standard_normal(1000) + 3, side="upper") > 0.95
    assert _run_gpd_p(np.random.standard_normal(1000) - 3, side="lower") > 0.95
    assert _run_gpd_p(np.random.standard_normal(1000) - 3, side="upper") < 0.05


test_run_gpd_p()


def get_bs_p(a, x=0, side="double", axis=None):
    """Fit tail with generalized pareto distribution to get p-value of x0.
    Based on Knijnenburg et al, 2009 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2687965/)
    Here we use a komogorov-smirnof test for equality of distributions 
    instead of Anderson-Darling as used in their paper. Optionally
    fit on an ndimensional array.
    Parameters
    ==========
    a: array
        Array of values containing the bootstrap or permutation distribution
    x0: float
        The value the distribution is being tested against
    side: string
        Specify the tail of the distribution to be tested must be one of ["upper", "lower"]
    axis: None or int
        Specifies the dimension along which the bootstraps/permutations are found
    Returns
    =======
    p: float or array
        Fitted p-value, returns a single value if axis is None
    """
    if axis is not None:
        new_shape = np.array(a.shape)[np.arange(len(a.shape)) != axis]
        a = a.reshape(a.shape[axis], -1).T
    else:
        a = a.reshape(1, -1)

    res = np.zeros(a.shape[0])
    for ii, aa in enumerate(a):
        if side == "double":
            res[ii] = (
                np.min((_run_gpd_p(aa, x, "upper"), _run_gpd_p(aa, x, "lower"))) * 2
            )
            if res[ii] > 1:
                res[ii] = 1
        elif side in ["upper", "lower"]:
            res[ii] = _run_gpd_p(aa, x, side)
        else:
            raise ValueError(
                f'side must be one of ["upper", "lower", "double"], you provided {side}'
            )
    if axis is None:
        return res[0]
    else:
        return res.T.reshape(new_shape)


def get_bs_stats(alpha, bs_series):
    qup = 1 - (alpha / 2)
    qdn = alpha / 2
    vrd = {}
    vrd[f"q_{qup:0.4f}"] = bs_series.quantile(qup)
    vrd[f"q_{qdn:0.4f}"] = bs_series.quantile(qdn)
    vrd["uncorrected_p"] = get_bs_p(bs_series.values, 0)
    vrd["sign"] = np.sign(bs_series.mean())
    vrd["mean_val"] = bs_series.mean()
    vrd["std_val"] = bs_series.std()
    return vrd


def get_bs_res(bs, alpha, side="double"):
    bs_res = {}
    bs_res["bs_mean"] = bs.mean(0).squeeze()
    bs_res["bs_std"] = bs.std(0).squeeze()
    bs_res["bs_sign"] = np.sign(bs_res["bs_mean"])
    ps = get_bs_p(bs, 0, side, axis=0).squeeze()
    bs_res["bs_uncorrected_p"] = ps
    bs_res["bs_p"] = multipletests(ps.flatten(), method="sidak")[1].reshape(ps.shape)
    bs_res["bs_sig"] = bs_res["bs_p"] < alpha

    bs_res["bs_signed_ps"] = bs_res["bs_p"] * bs_res["bs_sign"]

    return bs_res


def get_sig_var_difs(
    alpha,
    bs_var_res,
    bs_cb_var_res,
    varexs=["agh_varex", "mfg_varex", "model_varex", "site_varex"],
    pct_varexs=["variance"],
    verbose=False,
):
    bs_var_res = bs_var_res.copy(deep=True)
    bs_var_res = bs_var_res.merge(
        bs_cb_var_res.loc[:, ["metric", "pn"] + varexs + pct_varexs],
        how="left",
        on=["metric", "pn"],
        suffixes=["", "_cb"],
    )
    for var in varexs:
        bs_var_res[var + "_dif"] = bs_var_res[var] - bs_var_res[var + "_cb"]
    for pv in pct_varexs:
        bs_var_res[f"pct_{pv}_dif"] = (
            (bs_var_res[pv] - bs_var_res[pv + "_cb"]) / bs_var_res[pv] * 100
        )
    bs_dif_sig = []
    nmetrics = bs_var_res.metric.nunique()
    mi = 0
    for mm, df in bs_var_res.groupby("metric"):
        if verbose and mi == 0:
            print(f"Beginning processing metric {mi} of {nmetrics}")
        for var in varexs:
            vrd = get_bs_stats(alpha, df[var + "_dif"])
            vrd["mean_dif"] = vrd["mean_val"].copy()
            vrd["metric"] = mm
            vrd["level"] = var
            bs_dif_sig.append(vrd)
        for var in pct_varexs:
            vrd = get_bs_stats(alpha, df[f"pct_{var}_dif"])
            vrd["mean_dif"] = vrd["mean_val"].copy()
            vrd["metric"] = mm
            vrd["level"] = f"pct_{var}"
            bs_dif_sig.append(vrd)
        mi += 1
        if verbose and mi % 50 == 0:
            print(f"Completed {mi} metrics.")
    bs_dif_sig = pd.DataFrame(bs_dif_sig)
    return bs_var_res, bs_dif_sig


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


def get_cfns(df, name, include0=False):
    cfns = df.query("pn == 0 & name == @name").cfn.values
    cfns = np.array([c for c in cfns])
    perm_cfns = []
    if include0:
        perm_cfns.append(cfns)
    for pn in df.pn.unique():
        tmp = df.query("pn == @pn & name == @name").cfn.values
        perm_cfns.append(np.array([c for c in tmp]))

    perm_cfns = np.array(perm_cfns)
    return cfns, perm_cfns


def invert_lut(lut):

    inds = {v: [] for v in lut.values()}

    for k, v in lut.items():
        inds[v].append(k)
    inds = {k: np.array(v) for k, v in inds.items()}
    return inds


def collapse_cfn(cfns, inds, order):
    cfns_ut = cfns.copy()
    if len(cfns.shape) < 3:
        raise ValueError("cfns needs to have at least 3 dimensions")
    collapse_cfns_a = np.zeros((*cfns.shape[:-2], len(order), cfns.shape[-1]))
    collapse_cfns = np.zeros((*cfns.shape[:-2], len(order), len(order)))

    for mi, mdl in enumerate(order):
        set_slice = tuple(
            [slice(None) for cc in range(len(cfns.shape) - 2)] + [mi, slice(None)]
        )
        val_slice = tuple(
            [slice(None) for cc in range(len(cfns.shape) - 2)]
            + [inds[mdl], slice(None)]
        )
        collapse_cfns_a[set_slice] = cfns_ut[val_slice].sum(-2)
    for mi, mdl in enumerate(order):
        set_slice = tuple(
            [slice(None) for cc in range(len(cfns.shape) - 2)] + [slice(None), mi]
        )
        val_slice = tuple(
            [slice(None) for cc in range(len(cfns.shape) - 2)]
            + [slice(None), inds[mdl]]
        )
        collapse_cfns[set_slice] = collapse_cfns_a[val_slice].sum(-1)
    assert cfns_ut.sum() == collapse_cfns.sum()
    return collapse_cfns


def test_collapse():
    inds = {0: [0, 1], 1: [2]}
    order = [0, 1]
    cfns = np.array([[[[1, 5, 7], [1, 2, 4], [3, 0, 8]]]])
    verify = np.array([[[[9.0, 11.0], [3.0, 8.0]]]])

    collapsed = collapse_cfn(cfns, inds, order)
    assert (collapsed == verify).all()

    collapsed = collapse_cfn(cfns[0], inds, order)
    verify = verify[0]
    assert (collapsed == verify).all()


test_collapse()


def normalize_cfn(cfns):
    return cfns / cfns.sum(-1)[..., np.newaxis]


def get_collapsed_perms(cfns, perm_cfns, model_lut, model_order, mfg_lut, mfg_order):
    model_inds = invert_lut(model_lut)
    mfg_inds = invert_lut(mfg_lut)

    model_cfns = collapse_cfn(cfns, model_inds, model_order)
    mfg_cfns = collapse_cfn(cfns, mfg_inds, mfg_order)

    model_cfns_norm = normalize_cfn(model_cfns)
    mfg_cfns_norm = normalize_cfn(mfg_cfns)

    model_perm_cfns_norm = normalize_cfn(
        collapse_cfn(perm_cfns, model_inds, model_order)
    )
    mfg_perm_cfns_norm = normalize_cfn(collapse_cfn(perm_cfns, mfg_inds, mfg_order))

    model_cfn_signed_ps_norm = get_cfn_sig(model_cfns_norm, model_perm_cfns_norm)
    mfg_cfn_signed_ps_norm = get_cfn_sig(mfg_cfns_norm, mfg_perm_cfns_norm)

    return model_cfns, mfg_cfns, model_cfn_signed_ps_norm, mfg_cfn_signed_ps_norm


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
    **kwargs,
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
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, **kwargs)
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


def get_complete_perms(df, nperms=101, tfmri=False):
    if tfmri:
        df["label"] = df.contrast
    else:
        df["label"] = df.modality + "_" + df.ubermetric + "_" + df.metric

    df = get_varex(df)

    df["atlas"] = df.metric.str.split("__").str[0]

    zero_worked = df.groupby(["label"]).pn.unique().apply(lambda x: 0 in x)
    keep_labels = (
        df.groupby(["label"])
        .pn.nunique()[(df.groupby(["label"]).pn.nunique() >= nperms) & zero_worked]
        .index.values
    )

    df_keep = df[df.label.isin(keep_labels)]

    first_n_pns = (df_keep.groupby("pn")[["label"]].nunique() == zero_worked.sum())[
        (df_keep.groupby("pn")[["label"]].nunique() == zero_worked.sum())
    ].index.values[:nperms]
    df_keep = df_keep[df_keep.pn.isin(first_n_pns)]

    return df_keep
