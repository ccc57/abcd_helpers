# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

# abcd_table_path = "./ABCDstudyDEAP2.0/"
# data_dir = Path("data")

# # Define the directories that we're going to read data from
# data_dir = Path("data")
# swarm_dir = Path("swarm_dir_bs")
# # get_ipython().run_line_magic('matplotlib', 'inline')

# data_dir = Path("data")
# swarm_out_dir = Path("bootstrap_test/single_runs")
# data_out_dir = Path("bootstrap_test")


# # # Load Data
# # Either load consolidated permutation results or load individual
#   permuatation pkzs and consolidate them
# con_mod_res_path = data_out_dir / "con_mod_separate_100.pkz"
# var_res_path = data_out_dir / "var_res_separate_100.pkz"
# cb_var_res_path = data_out_dir / "cb_var_res_separate_100.pkz"


def load_abcd_table(path):
    table = pd.read_csv(path, skiprows=[1], header=0, sep="\t")
    labels = pd.read_csv(path, nrows=1, header=0, sep="\t")
    labels = labels.T.reset_index().rename(columns={"index": "name", 0: "doc"})
    return table, labels


def append_abcd_table(path, table=None, label=None, on=None):
    if table is None:
        return load_abcd_table(path)
    if on is None:
        on = ["subjectkey", "interview_date", "visit", "lmt_run"]
    new_table, new_label = load_abcd_table(path)
    # Find drop common columns that aren't in the set we're merging on
    drop_cols = label.loc[
        label.doc.isin(new_label.doc) & ~label.name.isin(on), "name"
    ].values
    new_table.drop(drop_cols, axis=1, inplace=True)
    new_label = new_label[~new_label.name.isin(drop_cols) & ~new_label.name.isin(on)]

    # Get table shapes for error checking
    ts = table.shape
    nts = new_table.shape
    ls = label.shape
    nls = new_label.shape

    # Merge the tables
    table = table.merge(new_table, how="outer", on=on)

    # Check for success
    assert table.shape[0] == ts[0] == nts[0]
    assert table.shape[1] == (ts[1] + nts[1] - len(on))

    # Merge labels
    label = pd.concat([label, new_label]).reset_index(drop=True)

    assert label.shape[0] == table.shape[1]

    return table, label


def load_task_melt_contrasts(paths, contrasts, task_name):
    task_t = None
    task_l = None
    for pp in paths:
        task_t, task_l = append_abcd_table(pp, table=task_t, label=task_l)

    # Rearrange columns
    new_col_order = pd.concat(
        [
            task_l.name[
                ~task_l.doc.str.split("for ")
                .str[1]
                .str.split(" contrast in")
                .str[0]
                .isin(contrasts)
            ],
            task_l.name[
                task_l.doc.str.split("for ")
                .str[1]
                .str.split(" contrast in")
                .str[0]
                .isin(contrasts)
            ],
        ]
    ).values
    task_t = task_t.loc[:, new_col_order]
    task_l = task_l.set_index("name").T.loc[:, new_col_order].T.reset_index()

    # Manipulate labels to get more information about them
    if task_name == "nb":
        task_l.loc[
            task_l.doc.str.split("for ").str[1].str.split(" in").str[0].isin(contrasts),
            "contrast",
        ] = (task_l.doc.str.split("for ").str[1].str.split(" in").str[0])
        task_l.loc[pd.notnull(task_l.contrast), "roi_code"] = (
            task_l.loc[pd.notnull(task_l.contrast), "name"].str.split("x").str[-1]
        )
        task_l.loc[pd.notnull(task_l.contrast), "hemisphere"] = (
            task_l.loc[pd.notnull(task_l.contrast), "roi_code"].str.split("g").str[0]
        )
        task_l.loc[pd.notnull(task_l.contrast), "roi_num"] = (
            task_l.loc[pd.notnull(task_l.contrast), "roi_code"]
            .str.split("p")
            .str[-1]
            .astype(int)
        )
        task_meta_cols = task_l.name[
            ~task_l.doc.str.split("for ").str[1].str.split(" in").str[0].isin(contrasts)
        ]
    else:
        task_l.loc[
            task_l.doc.str.split("for ")
            .str[1]
            .str.split(" contrast in")
            .str[0]
            .isin(contrasts),
            "contrast",
        ] = (task_l.doc.str.split("for ").str[1].str.split(" contrast in").str[0])
        task_l.loc[pd.notnull(task_l.contrast), "roi_code"] = (
            task_l.loc[pd.notnull(task_l.contrast), "name"].str.split("x").str[-1]
        )
        task_l.loc[pd.notnull(task_l.contrast), "hemisphere"] = (
            task_l.loc[pd.notnull(task_l.contrast), "roi_code"].str.split("g").str[0]
        )
        task_l.loc[pd.notnull(task_l.contrast), "roi_num"] = (
            task_l.loc[pd.notnull(task_l.contrast), "roi_code"]
            .str.split("p")
            .str[-1]
            .astype(int)
        )
        task_meta_cols = task_l.name[
            ~task_l.doc.str.split("for ")
            .str[1]
            .str.split(" contrast in")
            .str[0]
            .isin(contrasts)
        ]

    task_t_melt = []
    for contrast in contrasts:
        tmpdf = task_t.loc[:, task_meta_cols]
        tmpdf["contrast"] = contrast
        if task_name == "nb":
            contrast_cols = task_l.name[
                task_l.doc.str.split("for ").str[1].str.split(" in").str[0] == contrast
            ].values
        else:
            col_str = (
                task_l.doc.str.split("for ").str[1].str.split(" contrast in").str[0]
            )
            contrast_cols = task_l.name[col_str == contrast].values
        contrast_df = task_t.loc[:, contrast_cols]
        contrast_df.columns = [
            task_l.loc[task_l.name == cc, "roi_code"].values[0]
            for cc in contrast_df.columns.values
        ]
        task_t_melt.append(pd.concat([tmpdf, contrast_df], axis=1))
    task_t_melt = pd.concat(task_t_melt)
    task_t_melt["task"] = task_name
    return task_t_melt, task_l


def make_long(modality, labels, tables, new_id_vars=["modality", "metric"]):
    """
    Parameters
    ----------
    modality: string
        Name of the modality to extract and make long
    labels: DataFrame
        Information about the column names in the existing tables
        and the appropriate names for those columns in the long table
    tables: Dictonary of DataFrames with source file as key
        The tables from which information will be extracted
    new_id_vars: list of strings, default ['modality', 'metric']
        Names of new id variables described in labels that will
        be added to rows of long table

    Returns
    -------
    longtbl: DataFrame
        Partially melted dataframe that now has new_id_vars added

        """
    longtbl = []
    gb = labels.query("dat & modality == @modality").groupby(new_id_vars)
    for x, df in gb:
        # For each set of unique new_id_vars, extract the appropriate
        # information from each source table and merge them
        for sfi, sf in enumerate(np.unique(df.source_file.values)):
            # Combine the id columns present for this source file
            # with the value columns present in this group of new_id_vars
            new_cols = pd.concat(
                [
                    labels.query("source_file == @sf & ~dat"),
                    df.query("source_file == @sf"),
                ]
            )

            # Extract the mapping from orig column names to long names
            col_map = {
                nn: lfn
                for nn, lfn in new_cols.loc[:, ["name", "long_form_name"]].values
            }

            # Create map between new_id_var names and values
            id_var_map = {gbi: xi for gbi, xi in zip(new_id_vars, x)}

            # Pull the columns, rename them, add new_id_vars
            inner_tmp = (
                tables[sf]
                .loc[:, new_cols.name]
                .rename(columns=col_map)
                .assign(**id_var_map)
                .assign(source_file=sf)
            )

            if sfi == 0:
                tmp_df = inner_tmp
            else:
                # Figure out which columns are present to merge on
                merge_on = set(tmp_df.columns).intersection(inner_tmp.columns)
                # tbl_id and source_file depend on source_file
                # so don't merge on them
                merge_on.discard("tbl_id")
                merge_on.discard("source_file")
                merge_on.discard("dataset_id")
                # merge the tables
                tmp_df = tmp_df.merge(
                    inner_tmp, on=list(merge_on), how="outer", suffixes=("", "_y")
                )
        longtbl.append(tmp_df)

    longtbl = pd.concat(longtbl, sort=False, ignore_index=True)
    return longtbl


def bal_samp(
    df, strata, balance, order, keys, n_splits=5, n_draws=100, drop_strata=False
):
    """Balanced sampling across strata

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe from which you want to sample
    strata: str or list of str
        Name(s) of the column or columns that define the groups
        from which you want a balanced sample
    balance: str or list of str
        Name(s) of the columns or columns containing the factors
        you want to evenly sample across strata
    order: str or list of str
        Name(s) of the column whose distribution you want to preserve
    keys: list of str
        Name(s) of the column(s) that you will use to match
        the output back to your original column
    n_splits: int
        Number of cross validation folds you want to create per draw
    n_draws: int
        Number of balanced samples of your dataset you want to create

    Returns
    -------
    draws_df: pandas.DataFrame
        Dataframe with number of rows about equal to number of rows in df and
        number of columns equal to n_draws + len(keys) + len(strata) +
        len(balance) + len(order). Contains the crossfold labels for balanced
        sampling across the strata you've defined.
    """
    # create dict of minimum count in each strata of each combination of balance factor
    bal_dict = (
        df.groupby(strata + balance)[[keys[0]]]
        .nunique()
        .groupby(balance)
        .min()
        .to_dict("index")
    )
    bal_dict = {k: v[keys[0]] for k, v in bal_dict.items()}

    # Appologies for the disgusting nested loops
    # For each draw, at each strata level, for each unique combin
    draws_df = []
    # For each draw
    for nn in range(n_draws):
        strat_df = []
        # From each strata group
        for x, gbdf in df.groupby(strata):
            cvs_df = []
            # from each unique combination of balance values
            for bal_vals, num in bal_dict.items():
                # create an index selecting the rows at those balance values
                ind = np.ones((len(gbdf))).astype(bool)
                for bcol, bv in zip(balance, bal_vals):
                    ind = np.logical_and(ind, gbdf[bcol] == bv)
                if ind.sum() == 0:
                    if drop_strata:
                        print(
                            f"There are no rows with {bal_vals} for "
                            f"{balance} in group {x}. "
                        )

                        continue
                    else:
                        raise ValueError(
                            f"There are no rows with {bal_vals} for "
                            f"{balance} in group {x}. "
                        )

                # draw a random sample of the group members
                # that meet the balance criteria
                # and sort them by the order values
                bal_df = gbdf[ind].sample(n=num).sort_values(order).loc[:, keys]
                # create a list of the cross validation values long enough to match
                cv_inds = list(np.arange(n_splits)) * ((len(bal_df) // n_splits) + 1)
                bal_df["draw_%d" % nn] = cv_inds[: len(bal_df)]
                # and append them to a list
                cvs_df.append(bal_df)
            # combine these lists to get all the rows for that strata
            # and append them to create a list of selected rows from all the strata
            strat_df.append(pd.concat(cvs_df).loc[:, ["draw_%d" % nn]])
        # pull these all together to create the draws dataframe
        draws_df.append(pd.concat(strat_df))
    draws_df = pd.concat(draws_df, axis=1)
    # Merge back in the indicator variables
    draws_df = df.loc[:, keys + strata + balance + order].merge(
        draws_df, right_index=True, left_index=True, how="left"
    )
    # make sure the shape is still ok
    assert draws_df.shape[0] == df.shape[0]
    assert draws_df.shape[1] == (
        n_draws + len(keys) + len(strata) + len(balance) + len(order)
    )
    return draws_df


def gen_binned_perms(df, bin_levels, n_perms=1000, boot=False):
    df = df.copy(deep=True)
    permed_inds = []
    permed_inds.append(df.index.values)
    if "ind" not in df.columns:
        df["ind"] = df.index.values
    else:
        raise Exception
    for pn in range(n_perms):
        if not boot:
            permed_inds.append(
                df.groupby(bin_levels).ind.transform(np.random.permutation).values
            )
        else:
            permed_inds.append(df.sample(frac=1, replace=True).ind.values)
    return permed_inds


def get_cols(raw_df):
    base_meta_cols = [
        "contrast",
        "fmri_beta_gparc_numtrs",
        "fmri_beta_gparc_tr",
        "lmt_run",
        "mid_beta_seg_dof",
        "task",
        "collection_id",
        "dataset_id",
        "subjectkey",
        "src_subject_id",
        "interview_date",
        "interview_age",
        "gender",
        "event_name",
        "visit",
        "rsfm_tr",
        "eventname",
        "rsfm_nreps",
        "rsfm_numtrs",
        "pipeline_version",
        "scanner_manufacturer_pd",
        "scanner_type_pd",
        "mri_info_deviceserialnumber",
        "magnetic_field_strength",
        "procdate",
        "collection_title",
        "promoted_subjectkey",
        "study_cohort_name",
        "ehi_ss_score",
        "_merge",
        "qc_ok",
        "age_3mos",
        "abcd_betnet02_id",
        "fsqc_qc",
        "rsfmri_cor_network.gordon_visitid",
        "mrirscor02_id",
        "site_id_l",
        "mri_info_manufacturer",
        "mri_info_manufacturersmn",
        "mri_info_deviceserialnumber",
        "mri_info_magneticfieldstrength",
        "mri_info_softwareversion",
        "unique_scanner",
        "tbl_id",
        "tbl_visitid",
        "modality",
        "metric",
        "source_file",
        "tbl_id_y",
        "source_file_y",
        "run",
        "mri_info_visitid",
        "dmri_dti_postqc_qc",
        "iqc_t2_ok_ser",
        "iqc_mid_ok_ser",
        "iqc_sst_ok_ser",
        "iqc_nback_ok_ser",
        "tfmri_mid_beh_perform.flag",
        "tfmri_nback_beh_perform.flag",
        "tfmri_sst_beh_perform.flag",
        "tfmri_mid_all_beta_dof",
        "tfmri_mid_all_sem_dof",
        "tfmri_sst_all_beta_dof",
        "tfmri_sst_all_sem_dof",
        "tfmri_nback_all_beta_dof",
        "tfmri_nback_all_sem_dof",
        "mrif_score",
        "mrif_hydrocephalus",
        "mrif_herniation",
        "mr_findings_ok",
        "tbl_numtrs",
        "tbl_dof",
        "tbl_nvols",
        "tbl_tr",
        "tbl_subthresh.nvols",
        "rsfmri_cor_network.gordon_tr",
        "rsfmri_cor_network.gordon_numtrs",
        "rsfmri_cor_network.gordon_nvols",
        "rsfmri_cor_network.gordon_subthresh.nvols",
        "rsfmri_cor_network.gordon_subthresh.contig.nvols",
        "rsfmri_cor_network.gordon_ntpoints",
        "dataset_id_y",
        "tbl_mean.motion",
        "tbl_mean.trans",
        "tbl_mean.rot",
        "tbl_max.motion",
        "tbl_max.trans",
        "tbl_max.rot",
    ]
    meta_cols = raw_df.columns[raw_df.columns.isin(base_meta_cols)].values
    metric_cols = raw_df.columns[~raw_df.columns.isin(base_meta_cols)].values
    return metric_cols, meta_cols


def big_sites(size_limit, df, metric_cols):
    notnull_mask = np.logical_and(
        ~(df.loc[:, metric_cols].isnull().sum(1).astype(bool)),
        np.isfinite(df.loc[:, metric_cols]).sum(1).astype(bool),
    )
    notnull_mask = np.logical_and(notnull_mask, df.unique_scanner.notnull())
    notnull_mask = np.logical_and(notnull_mask, df.qc_ok == 1)
    # for right now, just focus on sites with more than size_limit scans
    scans_per_sn = (
        df.loc[notnull_mask, :]
        .groupby(["unique_scanner"])[["collection_id"]]
        .count()
        .sort_values("collection_id", ascending=False)
    )
    big_filter = df.unique_scanner.isin(
        scans_per_sn.query("collection_id > @size_limit").index
    )
    big_sns = df.loc[big_filter & notnull_mask, :]
    scans_per_bigsn = (
        big_sns.groupby("unique_scanner")[["collection_id"]]
        .count()
        .sort_values("collection_id", ascending=False)
    )

    print(
        f"Number of sites with more than {size_limit} scans:",
        len(big_sns.unique_scanner.unique()),
    )
    print(
        f"Number of scans collected at sites with more than {size_limit} scans:",
        len(big_sns),
    )
    print(
        f"Number of subjects collected at sites with more than {size_limit} scans:",
        big_sns.subjectkey.nunique(),
    )
    scans_per_bigsn.query("collection_id > @size_limit")
    return big_sns


def drop_bad(
    df, metric_cols, val_range=None, std_range=None, pct_range=None, verbose=False
):
    if verbose:
        df = df.copy()
        df["cr"] = df.contrast + df.run
        max_cr = df.groupby("subjectkey")[["cr"]].nunique().max()
        full_data_subs = len(
            pd.unique(
                (df.groupby("subjectkey")[["cr"]].nunique() == max_cr)
                .query("cr")
                .index.values
            )
        )

    notnull_mask = np.logical_and(
        ~(df.loc[:, metric_cols].isnull().sum(1).astype(bool)),
        np.isfinite(df.loc[:, metric_cols]).sum(1).astype(bool),
    )
    dropped_sum = (~notnull_mask).sum()
    if verbose:
        max_cr = df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique().max()
        dropped_subs = full_data_subs - len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == max_cr
                )
                .query("cr")
                .index.values
            )
        )
        print(f"{dropped_sum} contrasts dropped for null data in one of the metrics")
        print(
            f"{dropped_subs} subjects dropped for null data in one "
            f"metric on one contrast "
        )

        full_data_subs = len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == df.loc[notnull_mask, :]
                    .groupby("subjectkey")[["cr"]]
                    .nunique()
                    .max()
                )
                .query("cr")
                .index.values
            )
        )
    assert notnull_mask.sum() > 0

    notnull_mask = np.logical_and(notnull_mask, df.unique_scanner.notnull())
    if verbose:
        dropped_subs = full_data_subs - len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == df.loc[notnull_mask, :]
                    .groupby("subjectkey")[["cr"]]
                    .nunique()
                    .max()
                )
                .query("cr")
                .index.values
            )
        )
        print(
            f"{(~notnull_mask).sum() - dropped_sum} contrasts dropped for not having a scanner id"
        )
        print(f"{dropped_subs} subjects dropped for not having a scanner id")
        full_data_subs = len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == df.loc[notnull_mask, :]
                    .groupby("subjectkey")[["cr"]]
                    .nunique()
                    .max()
                )
                .query("cr")
                .index.values
            )
        )
    dropped_sum = (~notnull_mask).sum()
    assert notnull_mask.sum() > 0

    notnull_mask = np.logical_and(notnull_mask, df.qc_ok == 1)
    if verbose:
        dropped_subs = full_data_subs - len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == df.loc[notnull_mask, :]
                    .groupby("subjectkey")[["cr"]]
                    .nunique()
                    .max()
                )
                .query("cr")
                .index.values
            )
        )
        print(f"{(~notnull_mask).sum() - dropped_sum} contrasts dropped for failing QC")
        print(f"{dropped_subs} subjects dropped because one run failed QC")
        full_data_subs = len(
            pd.unique(
                (
                    df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                    == df.loc[notnull_mask, :]
                    .groupby("subjectkey")[["cr"]]
                    .nunique()
                    .max()
                )
                .query("cr")
                .index.values
            )
        )
    dropped_sum = (~notnull_mask).sum()
    assert notnull_mask.sum() > 0

    if val_range is not None:
        notnull_mask = np.logical_and(
            notnull_mask,
            (np.abs(df.loc[:, metric_cols]) < val_range).product(1).astype(bool),
        )
        if verbose:
            dropped_subs = full_data_subs - len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
            print(
                f"{~notnull_mask.sum - dropped_sum} contrasts dropped "
                f"for exceeding value range "
            )

            print(
                f"{dropped_subs} subjects dropped because one contrast "
                "had a value exceeding value range "
            )

            full_data_subs = len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
        dropped_sum = (~notnull_mask).sum()
        assert notnull_mask.sum() > 0

    if std_range is not None:
        notnull_mask = np.logical_and(
            notnull_mask,
            (
                np.abs(
                    (df.loc[:, metric_cols] - df.loc[:, metric_cols].mean())
                    / df.loc[:, metric_cols].std()
                )
                < std_range
            )
            .product(1)
            .astype(bool),
        )
        if verbose:
            dropped_subs = full_data_subs - len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
            print(
                f"{(~notnull_mask).sum() - dropped_sum} contrasts dropped for exceeding variance range"
            )
            print(
                f"{dropped_subs} subjects dropped because one contrast had a value exceeding variance range"
            )
            full_data_subs = len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
        dropped_sum = (~notnull_mask).sum()
        assert notnull_mask.sum() > 0

    if pct_range is not None:
        top_pct = (100 - pct_range) / 100
        bottom_pct = pct_range / 100
        notnull_mask = np.logical_and(
            notnull_mask,
            (
                df.loc[:, metric_cols]
                <= df.loc[:, metric_cols].quantile([top_pct]).iloc[0, :]
            )
            .product(1)
            .astype(bool),
        )
        notnull_mask = np.logical_and(
            notnull_mask,
            (
                df.loc[:, metric_cols]
                >= df.loc[:, metric_cols].quantile([bottom_pct]).iloc[0, :]
            )
            .product(1)
            .astype(bool),
        )
        if verbose:
            dropped_subs = full_data_subs - len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
            print(
                f"{~notnull_mask.sum - dropped_sum} contrasts dropped "
                f"for exceeding percentile range "
            )

            print(
                f"{dropped_subs} subjects dropped because one contrast "
                "had a value exceeding percentile range "
            )

            full_data_subs = len(
                pd.unique(
                    (
                        df.loc[notnull_mask, :].groupby("subjectkey")[["cr"]].nunique()
                        == df.loc[notnull_mask, :]
                        .groupby("subjectkey")[["cr"]]
                        .nunique()
                        .max()
                    )
                    .query("cr")
                    .index.values
                )
            )
        dropped_sum = (~notnull_mask).sum()
        assert notnull_mask.sum() > 0
    return df.loc[notnull_mask, :].copy(deep=True)


def get_full_subjects(longtbl, modality, metric_range=None, **kwargs):

    for ii, mm in enumerate(longtbl.query("modality == @modality").metric.unique()):
        metric_df = longtbl.loc[
            ((longtbl.modality == modality) & (longtbl.metric == mm))
        ].copy()
        unused_cols = metric_df.columns[(metric_df.notnull().sum() == 0)].values
        metric_df.drop(unused_cols, axis=1, inplace=True)

        metric_cols, meta_cols = get_cols(metric_df)

        if metric_range is not None:
            try:
                metric_nb = drop_bad(
                    metric_df, metric_cols, val_range=metric_range[mm], **kwargs
                )
            except KeyError:
                metric_nb = drop_bad(metric_df, metric_cols, **kwargs)
        else:
            metric_nb = drop_bad(metric_df, metric_cols, **kwargs)
        if ii == 0:
            full_subj = set(metric_nb.subjectkey.unique())
        else:
            full_subj = full_subj.intersection(metric_nb.subjectkey.unique())

    return full_subj


def make_cmd_list(
    user,
    prange,
    modality,
    metrics,
    bundle=5,
    n_draws=25,
    run=None,
    task=None,
    contrast=None,
    out_base="/data/nielsond/abcd/nielson_abcd_2018/",
    swarm_dir="/data/nielsond/abcd/nielson_abcd_2018/swarm_dir",
):
    if (modality == "tfmri") & ((run is None) | (task is None) | (contrast is None)):
        raise ValueError(
            "If modality is tfmri, then run, task and contrast mus be provided"
        )
    swarm_dir = Path(swarm_dir)
    cmds = []
    job_n = 0
    rsync_in = (
        ""
    )  #'rsync -avch /data/MLDSST/abcd/abcd_swarm_dir /lscratch/$SLURM_JOBID/; '
    cmd = rsync_in
    for pn in range(*prange):
        for metric in metrics:
            if modality != "tfmri":
                out_path = f"{out_base}/release11/pn-{pn:04d}_{modality}_{metric}.pkz"
            else:
                out_path = (
                    f"{out_base}/release11/pn-{pn:04d}_crt_{task}_{contrast}_{run}.pkz"
                )

            if not Path(out_path).exists():

                perm_file = swarm_dir / f"perms_{modality}.pkz"
                ymap_file = swarm_dir / "yfit.pkz"
                if modality != "tfmri":
                    input_file = swarm_dir / f"{modality}_{metric}.pkz"
                else:
                    input_file = swarm_dir / f"crt_{task}_{contrast}_{run}_beta.pkz "

                cmd_base = (
                    'export SINGULARITY_BINDPATH="/gs3,/gs4,/gs5,/gs6,/gs7,/gs8,/gs9,/gs10,/gs11,/spin1,/scratch,/fdb,/data";'
                    + " singularity exec -H ~/temp_for_singularity /data/MLDSST/singularity_images/abcd_tmp-2018-04-26-a7f5b7d6a3b4.img"
                    + f" python {swarm_dir}/run_abcd_perm_new_draws.py {pn} {perm_file}"
                    + f" {ymap_file} {input_file}"
                    + f" {out_path} 30 --n_draws={n_draws}; "
                )
                # rsync_out = f'rsync -ach {out_path} {local_path}; chown $USER:MLDSST {local_path}; '
                cmd += cmd_base  # + rsync_out
                job_n += 1
                if job_n == bundle:
                    cmds.append(cmd)
                    cmd = rsync_in
                    job_n = 0
    return cmds


def calc_draw_mis(ab_draws_df):
    draw_labels = [
        "draw_0",
        "draw_1",
        "draw_2",
        "draw_3",
        "draw_4",
        "draw_5",
        "draw_6",
        "draw_7",
        "draw_8",
        "draw_9",
        "draw_10",
        "draw_11",
        "draw_12",
        "draw_13",
        "draw_14",
        "draw_15",
        "draw_16",
        "draw_17",
        "draw_18",
        "draw_19",
        "draw_20",
        "draw_21",
        "draw_22",
        "draw_23",
        "draw_24",
    ]
    mis = []
    for ii, dli in enumerate(draw_labels):
        for jj, dlj in enumerate(draw_labels[(ii + 1) :]):
            mir = {}
            mir["draw_a"] = dli
            mir["draw_b"] = dlj
            mir["mi"] = normalized_mutual_info_score(
                (ab_draws_df[dli] == 1).values, (ab_draws_df[dlj] == 1).values
            )
            mis.append(mir)
    mis = pd.DataFrame(mis)
    return mis


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


# TODO: these functions were commented out in the notebooks:
# def adjr2_to_r2(y, n, p):
#     """Converts adjusted R2 to unadjusted R2
#     Based on formula here: https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

#     Parameters
#     ----------
#     y: float
#         adjusted R2 you'd like to convert
#     n: int
#         number of samples
#     p: int
#         number of parameters

#     Returns
#     -------
#     res: float
#         adjusted R2
#     """
#     res = (((n - 1) * y) + (p * (-1 * y)) + p)/ (n - 1)
#     return res

# def get_unadj_r2(df, n):
#     df['int_r2'] = adjr2_to_r2(df.int_r2, n, 1)
#     df['agh_r2'] = adjr2_to_r2(df.agh_r2, n, 4)
#     df['aghs_r2'] = adjr2_to_r2(df.aghs_r2, n, 5)
#     df['aghss_r2'] = adjr2_to_r2(df.aghss_r2, n, 6)
#     df['aghsss_r2'] = adjr2_to_r2(df.aghsss_r2, n, 7)
#     return df
