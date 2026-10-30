# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from thicket import Thicket, EmptyMetadataTable
from test_concat_thickets import test_concat_thickets_columns
from utils import check_identity


def check_groupby(th, columns_values):
    """Check groupby function for Thicket object.

    Arguments:
        th (Thicket): Thicket object to test.
        columns_values (dict): Dictionary of columns to group by.
    """
    # inspect all use cases
    for column in columns_values:
        unique_values = sorted(th.metadata[column].unique().tolist())
        th_list = list(th.groupby(column).values())

        for thicket in th_list:
            check_identity(th, thicket, ["default_metric", "profile_idx_name"])

        # inspect all unique values in the use case
        for itr, uni_val in enumerate(unique_values):
            # expected profile hashes and nodes
            exp_profiles = sorted(
                th.metadata.index[th.metadata[column] == uni_val].tolist()
            )
            exp_nodes = sorted(
                th.dataframe[th.dataframe.index.get_level_values(1).isin(exp_profiles)]
                .index.get_level_values(0)
                .drop_duplicates()
                .tolist()
            )

            # metadata profile hash for unique value sub-thicket
            metadata_profiles = sorted(th_list[itr].metadata.index.tolist())
            assert metadata_profiles == exp_profiles

            #  performance data table profile hash for unique value sub-thicket
            ensemble_profiles = sorted(
                th_list[itr]
                .dataframe.index.get_level_values(1)
                .drop_duplicates()
                .tolist()
            )
            assert ensemble_profiles == exp_profiles

            # performance data table nodes for unique value sub-thicket
            ensemble_nodes = sorted(
                th_list[itr]
                .dataframe.index.get_level_values(0)
                .drop_duplicates()
                .tolist()
            )
            assert ensemble_nodes == exp_nodes

            # aggregated statistics nodes for unique value sub-thicket
            stats_nodes = sorted(
                th_list[itr]
                .statsframe.dataframe.index.get_level_values(0)
                .drop_duplicates()
                .tolist()
            )
            assert stats_nodes == exp_nodes

            # check for name column in aggregated statistics table
            assert "name" in th_list[itr].statsframe.dataframe.columns

    # drop all rows of the metadata table
    th.metadata = th.metadata.iloc[0:0]

    # check for empty metadata table exception
    with pytest.raises(EmptyMetadataTable):
        th.groupby(["user"])


def test_aggregate(rajaperf_cuda_block128_1M_cali, intersection, fill_perfdata):
    tk = Thicket.from_caliperreader(
        rajaperf_cuda_block128_1M_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    gb = tk.groupby("spot.format.version")

    epsilon = 0.000001

    # Variance of "Min time/rank" for the "Algorithm" node over the 10 profiles
    population_var = 1.825684e-08  # NumPy semantics (ddof=0), e.g., np.var
    sample_var = 2.0285377777777793e-08  # pandas semantics (ddof=1), e.g., "var"

    def _check_values(_tk_agg, expected_var):
        node = [
            node
            for node in _tk_agg.dataframe.index.get_level_values("node")
            if node.frame["name"] == "RAJAPerf"
        ][0]
        assert (
            abs(_tk_agg.dataframe.loc[node, 2]["Min time/rank_mean"] - 1.7765899)
            < epsilon
        )

        algorithm_node = [
            node
            for node in _tk_agg.dataframe.index.get_level_values("node")
            if node.frame["name"] == "Algorithm"
        ][0]
        assert _tk_agg.dataframe.loc[algorithm_node, 2][
            "Min time/rank_var"
        ] == pytest.approx(expected_var, rel=1e-6)

    # Functions are passed to pandas as-is, so np.var computes the population variance
    tk_agg = gb.agg(
        func={"Min time/rank": [np.mean, np.var], "Total time": np.mean},
        disable_tqdm=True,
    )
    _check_values(tk_agg, population_var)
    tk_agg = gb.agg(func=[np.mean, np.var], disable_tqdm=True)
    _check_values(tk_agg, population_var)

    # pandas aggregation names use pandas semantics and name the columns the same way
    tk_agg = gb.agg(
        func={"Min time/rank": ["mean", "var"], "Total time": "mean"},
        disable_tqdm=True,
    )
    _check_values(tk_agg, sample_var)
    tk_agg = gb.agg(func=["mean", "var"], disable_tqdm=True)
    _check_values(tk_agg, sample_var)


def test_groupby(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    # example thicket
    th = Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    # use cases for string, numeric, and single value columns
    columns_values = ["user", "launchdate", "cali.channel"]

    check_groupby(th, columns_values)


def test_groupby_concat_thickets_columns(
    rajaperf_seq_O3_1M_cali, intersection, fill_perfdata
):
    """Tests case where the Sub-Thickets of a groupby are used in a columnar join"""
    # example thicket
    th = Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    # Creates four Sub-Thickets
    column = "unique_col"
    th.metadata[column] = [1, 2, 3, 4]
    th_list = list(th.groupby(column).values())

    thickets = [th_list[0], th_list[1], th_list[2], th_list[3]]
    thickets_cp = [
        th_list[0].deepcopy(),
        th_list[1].deepcopy(),
        th_list[2].deepcopy(),
        th_list[3].deepcopy(),
    ]

    selected_column = "ProblemSizeRunParam"
    combined_th = Thicket.concat_thickets(
        thickets=thickets,
        axis="columns",
        metadata_key=selected_column,
        disable_tqdm=True,
    )

    test_concat_thickets_columns((thickets, thickets_cp, combined_th))


def test_groupby_concat_thickets_columns_subthickets(
    rajaperf_seq_O3_1M_cali, intersection, fill_perfdata
):
    """Tests case where some specific Sub-Thickets of a groupby are used in a columnar join"""
    # example thicket
    th = Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    # Creates four Sub-Thickets
    column = "unique_col"
    th.metadata[column] = [1, 2, 3, 4]
    th_list = list(th.groupby(column).values())

    # Pick two Sub-Thickets to test if metadata and profile information is setup correctly
    selected_column = "ProblemSizeRunParam"
    problem_size = 10
    th_list[0].metadata[selected_column] = problem_size
    th_list[1].metadata[selected_column] = problem_size

    thickets = [th_list[0], th_list[1]]
    thickets_cp = [
        th_list[0].deepcopy(),
        th_list[1].deepcopy(),
    ]

    combined_th = Thicket.concat_thickets(
        thickets=thickets,
        axis="columns",
        metadata_key=selected_column,
        disable_tqdm=True,
    )

    test_concat_thickets_columns((thickets, thickets_cp, combined_th))
