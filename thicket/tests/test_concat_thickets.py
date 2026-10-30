# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re

import hatchet as ht
import pandas as pd
import pytest

from test_filter_metadata import filter_one_column
from test_filter_metadata import filter_multiple_and
from test_filter_stats import check_filter_stats
from test_query import check_query
from thicket import Thicket
from thicket.utils import DuplicateIndexError


def test_concat_thickets_index(mpi_scaling_cali, intersection, fill_perfdata):
    th_27 = Thicket.from_caliperreader(
        mpi_scaling_cali[0],
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    th_64 = Thicket.from_caliperreader(
        mpi_scaling_cali[1],
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    if intersection:
        calltree = "intersection"
    else:
        calltree = "union"
    tk = Thicket.concat_thickets(
        [th_27, th_64],
        calltree=calltree,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    # Check specific values. Row order can vary so use "sum" to check
    node = tk.dataframe.index.get_level_values("node")[8]
    assert sum(tk.dataframe.loc[node, "Min time/rank"]) == 0.000453

    # Check error thrown
    with pytest.raises(
        DuplicateIndexError,
    ):
        Thicket.from_caliperreader(
            [mpi_scaling_cali[0], mpi_scaling_cali[0]],
            node_ordering=False,
            intersection=intersection,
            fill_perfdata=fill_perfdata,
            disable_tqdm=True,
        )


def test_concat_thickets_columns(thicket_axis_columns):
    thickets, thickets_cp, combined_th = thicket_axis_columns
    # Check no original objects modified
    for i in range(len(thickets)):
        assert thickets[i].dataframe.equals(thickets_cp[i].dataframe)
        assert thickets[i].metadata.equals(thickets_cp[i].metadata)

    # Check dataframe shape. Should be columnar-joined
    assert combined_th.dataframe.shape[0] <= sum(
        [th.dataframe.shape[0] for th in thickets]
    )  # Rows. Should be <= because some rows will exist across multiple thickets.
    assert (
        combined_th.dataframe.shape[1]
        == sum([th.dataframe.shape[1] for th in thickets]) - len(thickets) + 1
    )  # Columns. (-1) for each name column removed, (+1) singular name column created.

    # Check metadata shape. Should be columnar-joined
    assert combined_th.metadata.shape[0] == max(
        [th.metadata.shape[0] for th in thickets]
    )  # Rows. Should be max because all rows should exist in all thickets.
    assert combined_th.metadata.shape[1] == sum(
        [th.metadata.shape[1] for th in thickets]
    ) - len(
        thickets
    )  # Columns. (-1) Since we added an additional column "ProblemSize".

    # Check profiles
    assert len(combined_th.profile) == sum([len(th.profile) for th in thickets])

    # Check profile_mapping
    assert len(combined_th.profile_mapping) == sum(
        [len(th.profile_mapping) for th in thickets]
    )

    # PerfData and StatsFrame nodes should be in the same order.
    assert (
        pd.unique(combined_th.dataframe.reset_index()["node"])
        == pd.unique(combined_th.statsframe.dataframe.reset_index()["node"])
    ).all()


def test_filter_concat_thickets_columns(thicket_axis_columns):
    thickets, thickets_cp, combined_th = thicket_axis_columns
    # columns and corresponding values to filter by
    columns_values = {
        ("default", "variant"): ["Base_Seq"],
        ("block_128", "cali.caliper.version"): ["2.9.0-dev"],
    }

    filter_one_column(combined_th, columns_values)
    filter_multiple_and(combined_th, columns_values)


def test_filter_stats_concat_thickets_columns(thicket_axis_columns, intersection):
    thickets, thickets_cp, combined_th = thicket_axis_columns

    columns_values = {
        ("test", "test_string_column"): ["less than 20"],
        ("test", "test_numeric_column"): [4, 15],
    }
    # set string column values
    less_than_20 = ["less than 20"] * 21
    less_than_45 = ["less than 45"] * 25

    if intersection:
        less_than_65 = ["less than 65"] * 18
        new_col = less_than_20 + less_than_45 + less_than_65
        combined_th.statsframe.dataframe[("test", "test_string_column")] = new_col
        # set numeric column values
        combined_th.statsframe.dataframe[("test", "test_numeric_column")] = range(0, 64)
    else:
        less_than_75 = ["less than 75"] * 28
        new_col = less_than_20 + less_than_45 + less_than_75
        combined_th.statsframe.dataframe[("test", "test_string_column")] = new_col
        # set numeric column values
        combined_th.statsframe.dataframe[("test", "test_numeric_column")] = range(0, 74)

    check_filter_stats(combined_th, columns_values)


def test_query_concat_thickets_columns(thicket_axis_columns, intersection):
    thickets, thickets_cp, combined_th = thicket_axis_columns
    # test arguments
    if intersection:
        # Shorter graph for intersection
        hnids = [
            0,
            1,
            2,
            3,
            4,
        ]
    else:
        hnids = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]  # "0" because top-level node "RAJAPerf" will be included in query result.
    query = (
        ht.QueryMatcher()
        .match("*")
        .rel(
            ".",
            lambda row: row["name"]
            .apply(lambda x: re.match(r"Algorithm*", x) is not None)
            .all(),
        )
    )

    check_query(combined_th, hnids, query)


def test_filter_profile_concat_thickets_columns(thicket_axis_columns):
    thickets, thickets_cp, combined_th = thicket_axis_columns

    rm_profs = [
        (1048576.0, "default"),
        (1048576.0, "block_128"),
        (1048576.0, "block_256"),
    ]
    keep_profs = [
        (2097152.0, "block_256"),
        (2097152.0, "default"),
        (2097152.0, "block_128"),
    ]

    tk_filt = combined_th.filter_profile(keep_profs)

    for component in [tk_filt.profile, tk_filt.profile_mapping.keys()]:
        assert all([prof not in component for prof in rm_profs])
        assert all([prof in component for prof in keep_profs])

    assert 1048576.0 not in tk_filt.dataframe.index.get_level_values(
        "ProblemSizeRunParam"
    )
    assert 2097152.0 in tk_filt.dataframe.index.get_level_values("ProblemSizeRunParam")
    assert 1048576.0 not in tk_filt.metadata.index
    assert 2097152.0 in tk_filt.metadata.index
