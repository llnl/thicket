# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import copy
import re

import caliperreader as cr
import hatchet as ht
import numpy as np
import pytest

import thicket as tt


def test_from_timeseries_cxx(example_timeseries_cxx):
    """Sanity test a thicket timeseries object"""
    th = tt.Thicket.from_timeseries(example_timeseries_cxx)

    # Check the object type
    assert isinstance(th, tt.Thicket)

    # Check the resulting dataframe shape
    assert th.dataframe.shape == (20, 17)
    assert "loop.start_iteration" not in th.dataframe.columns

    # Check a value in the dataframe
    assert (
        th.dataframe.loc[
            th.dataframe.index.get_level_values(0)[0], "avg#time.duration.ns"
        ].values[0]
        == 130295.0
    )


def test_from_timeseries_lulesh(example_timeseries):
    """Sanity test a thicket timeseries object"""
    th = tt.Thicket.from_timeseries(example_timeseries)

    # Check the object type
    assert isinstance(th, tt.Thicket)

    # Check the resulting dataframe shape
    assert th.dataframe.shape == (950, 18)
    assert "loop.start_iteration" not in th.dataframe.columns

    # Check a value in the dataframe
    assert (
        th.dataframe.loc[
            th.dataframe.index.get_level_values(0)[0], "alloc.region.highwatermark"
        ].values[0]
        == 25824351.0
    )


def test_timeseries_statsframe(example_timeseries):
    """Test the creation of a statsframe with timeseries thicket"""

    th = tt.Thicket.from_timeseries(example_timeseries)

    # Check that the aggregated statistics table is a Hatchet GraphFrame.
    assert isinstance(th.statsframe, ht.GraphFrame)
    # Check that 'name' column is in dataframe. If not, tree() will not work.
    assert "name" in th.statsframe.dataframe
    # Check length of graph is the same as the dataframe.
    assert len(th.statsframe.graph) == len(th.statsframe.dataframe)

    tt.stats.mean(th, columns=["alloc.region.highwatermark"])

    assert "alloc.region.highwatermark_mean" in th.statsframe.dataframe.columns
    assert (
        "alloc.region.highwatermark_mean"
        in th.statsframe.exc_metrics + th.statsframe.inc_metrics
    )
    assert "alloc.region.highwatermark_mean" in th.statsframe.show_metric_columns()

    # Expected tree output
    tree_output = th.statsframe.tree(metric_column="alloc.region.highwatermark_mean")

    # Check if tree output is correct.
    assert bool(re.search("63353240.000.*lulesh.cycle", tree_output))


def test_timeseries_temporal_pattern(mem_power_timeseries):
    """
    Test the stats temporal pattern calculation with timeseries thicket, should add new columns to stats frame and score appropriately
    """

    th = tt.Thicket.from_timeseries(mem_power_timeseries)

    returned_cols = tt.stats.calc_temporal_pattern(
        th, columns=["memstat.vmrss", "variorum.val.power_node_watts"]
    )

    # Check that the aggregated statistics table is a Hatchet GraphFrame.
    assert isinstance(th.statsframe, ht.GraphFrame)

    expected_cols = [
        "memstat.vmrss_pattern",
        "memstat.vmrss_temporal_score",
        "variorum.val.power_node_watts_pattern",
        "variorum.val.power_node_watts_temporal_score",
    ]
    for col in expected_cols:
        # Check that expected columns are in statsframe dataframe.
        assert col in th.statsframe.dataframe
        # Check the returned columns from calc_temporal_pattern.
        assert col in returned_cols

    # Check some values in the memory pattern column
    assert th.statsframe.dataframe["memstat.vmrss_pattern"].iloc[0] == "none"
    assert th.statsframe.dataframe["memstat.vmrss_pattern"].iloc[1] == "constant"
    assert (
        th.statsframe.dataframe["variorum.val.power_node_watts_pattern"].iloc[0]
        == "none"
    )
    assert (
        th.statsframe.dataframe["variorum.val.power_node_watts_pattern"].iloc[1]
        == "constant"
    )

    mem_tree = th.statsframe.tree(
        metric_column="memstat.vmrss_temporal_score",
        annotation_column="memstat.vmrss_pattern",
    )
    pow_tree = th.statsframe.tree(
        metric_column="variorum.val.power_node_watts_temporal_score",
        annotation_column="variorum.val.power_node_watts_pattern",
    )

    # Check if tree output is correct.
    assert bool(re.search("0.000", mem_tree))
    assert bool(re.search("0.013", pow_tree))


@pytest.mark.parametrize("level", ["loop.start_iteration", "iteration#lulesh.cycle"])
def test_timeseries_missing_observations(mem_power_timeseries, level):
    """Missing observations stay NaN, including before a node's first sample."""
    reader = cr.CaliperReader()
    reader.read(mem_power_timeseries)
    records = []
    for record in reader.records:
        record = record.copy()
        iteration = record.get("loop.start_iteration")
        if iteration is not None:
            record[level] = record.pop("loop.start_iteration")
        if iteration == "40":
            record["memstat.vmrss"] = "0"
        records.append(record)
        if iteration in ("20", "80"):
            optional = record.copy()
            optional["path"] = ["main", "optional"]
            optional["memstat.vmrss"] = "0" if iteration == "20" else "743"
            records.append(optional)
    reader.records = records
    original_records = copy.deepcopy(records)

    th = tt.Thicket.from_timeseries(reader, level=level)
    nodes = {node.frame["name"]: node for node in th.graph.traverse()}
    expected_values = {
        "main": [679, np.nan, np.nan, np.nan, np.nan],
        "lulesh.cycle": [743, 743, 0, 743, 743],
        "optional": [np.nan, 0, np.nan, np.nan, 743],
    }
    for name, expected in expected_values.items():
        values = th.dataframe.xs(nodes[name], level="node")["memstat.vmrss"]
        assert list(values.index) == [0, 20, 40, 60, 80]
        np.testing.assert_equal(values.to_numpy(), expected)
    assert reader.records == original_records
