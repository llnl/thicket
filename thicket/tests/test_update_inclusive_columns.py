# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""Tests of update_inclusive_columns, which Thicket inherits from Hatchet's
GraphFrame, on the MultiIndex dataframes of a Thicket.

The rows of a Thicket's performance data are indexed by "node", usually
"profile", and any other index levels of the original GraphFrames, such as
"rank". After concat_thickets(axis="columns"), the columns are a MultiIndex too.
Inclusive values must be computed separately for each combination of the index
levels other than "node".
"""

import numpy as np
import pandas as pd
from hatchet import GraphFrame
from hatchet.frame import Frame
from hatchet.graph import Graph
from hatchet.node import Node

from thicket import Thicket
from thicket.query import Query

# A -> B -> C, A -> B -> D, A -> E
TREE = [("A", "B"), ("B", "C"), ("B", "D"), ("A", "E")]
TREE_TIMES = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}


def _make_graphframe(edges, times):
    """Build a GraphFrame by hand, so that its graph can be any DAG.

    Arguments:
        edges (list): (parent, child) pairs of node names
        times (dict): maps a node name, or a (name, rank) tuple, to the exclusive
            "time" of that row

    Returns:
        (GraphFrame): GraphFrame indexed by "node", or by "node" and "rank"
    """
    keys = [key if isinstance(key, tuple) else (key,) for key in times]
    names = sorted({key[0] for key in keys})
    nodes = {name: Node(Frame({"name": name, "type": "function"})) for name in names}
    for parent, child in edges:
        nodes[parent].add_child(nodes[child])
        nodes[child].add_parent(nodes[parent])
    graph = Graph([node for node in nodes.values() if not node.parents])
    # Nodes hash by their ID, so set the IDs before indexing by node
    graph.enumerate_traverse()

    index_names = ["node", "rank"][: len(keys[0])]
    rows = [
        (nodes[key[0]],) + key[1:] + (key[0], time)
        for key, time in zip(keys, times.values())
    ]
    dataframe = pd.DataFrame(rows, columns=index_names + ["name", "time"])
    dataframe = dataframe.set_index(index_names).sort_index()
    return GraphFrame(graph, dataframe, exc_metrics=["time"], inc_metrics=[])


def _concat_profiles(graphframes, **kwargs):
    """Make each GraphFrame a profile of one Thicket.

    The metadata column "run" is the position of the profile's GraphFrame in
    graphframes. Keyword arguments are passed to Thicket.concat_thickets.
    """
    thickets = []
    for run, gf in enumerate(graphframes):
        tk = Thicket.thicketize_graphframe(gf, "run {}".format(run))
        # thicketize_graphframe leaves the metadata table empty, which squash()
        # rejects because its profiles don't match the performance data's
        tk.metadata = pd.DataFrame(
            {"run": [run]}, index=pd.Index(tk.profile, name=tk.profile_idx_name)
        )
        thickets.append(tk)
    return Thicket.concat_thickets(thickets, disable_tqdm=True, **kwargs)


def _scaled(values, factor):
    return {key: factor * value for key, value in values.items()}


def _assert_column(tk, column, *runs):
    """Check a column of the performance data of a Thicket from _concat_profiles.

    Arguments:
        tk (Thicket): Thicket to check
        column (str): column of tk.dataframe
        runs (dict): for each run, a dict that maps a node name, or a (name, rank)
            tuple, to the expected value. The rows of the column must be exactly
            the rows given.
    """
    expected = {}
    for run, values in enumerate(runs):
        for key, value in values.items():
            key = key if isinstance(key, tuple) else (key,)
            expected[(key[0], run) + key[1:]] = value

    run_of_profile = tk.metadata["run"]
    index = [
        (node.frame["name"], run_of_profile[profile]) + tuple(rest)
        for node, profile, *rest in tk.dataframe.index
    ]
    actual = pd.Series(
        tk.dataframe[column].to_numpy(), index=pd.MultiIndex.from_tuples(index)
    )
    pd.testing.assert_series_equal(
        actual.sort_index(), pd.Series(expected).sort_index(), check_names=False
    )


def _reference_inclusive(tk, exc_metric):
    """Compute the inclusive values of an exclusive metric independently.

    The inclusive value of a row is the sum of exc_metric over the row's node and
    all of its descendants, each counted once, over the rows that have the same
    values as the row in all other index levels (e.g., the same profile). NaNs
    are skipped, but a sum of only NaNs is NaN.
    """
    exc = tk.dataframe[exc_metric]
    nodes = exc.index.get_level_values("node")
    other_levels = [level for level in exc.index.names if level != "node"]
    parts = []
    for node in tk.graph.traverse():
        subgraph = exc[nodes.isin(list(node.traverse()))]
        sums = subgraph.groupby(level=other_levels).sum(min_count=1)
        own_rows = exc[nodes == node].index
        parts.append(
            pd.Series(
                sums.reindex(own_rows.droplevel("node")).to_numpy(), index=own_rows
            )
        )
    return pd.concat(parts).reindex(exc.index)


def _check_against_reference(tk):
    """Run update_inclusive_columns on tk and compare each inclusive column with
    _reference_inclusive."""
    exc_metrics = list(tk.exc_metrics)
    assert exc_metrics

    tk.update_inclusive_columns()

    for exc_metric in exc_metrics:
        if isinstance(exc_metric, tuple):  # concat_thickets(axis="columns")
            inc_metric = exc_metric[:-1] + (exc_metric[-1] + " (inc)",)
        else:
            inc_metric = exc_metric + " (inc)"
        assert inc_metric in tk.inc_metrics
        pd.testing.assert_series_equal(
            tk.dataframe[inc_metric],
            _reference_inclusive(tk, exc_metric),
            check_names=False,
            rtol=1e-12,
        )


def test_update_inclusive_columns_per_profile():
    """Each profile's inclusive values only sum that profile's exclusive values."""
    tk = _concat_profiles(
        [
            _make_graphframe(TREE, TREE_TIMES),
            _make_graphframe(TREE, _scaled(TREE_TIMES, 10)),
        ]
    )
    assert tk.dataframe.index.names == ["node", "profile"]
    exclusive = tk.dataframe["time"].copy()

    tk.update_inclusive_columns()

    inclusive = {"A": 15.0, "B": 9.0, "C": 3.0, "D": 4.0, "E": 5.0}
    _assert_column(tk, "time (inc)", inclusive, _scaled(inclusive, 10))
    assert tk.exc_metrics == ["time"]
    assert tk.inc_metrics == ["time (inc)"]
    # The exclusive column and the rows are unchanged
    pd.testing.assert_series_equal(tk.dataframe["time"], exclusive)

    # Inclusive values are recomputed from the exclusive values, so calling the
    # function again does not change them
    tk.update_inclusive_columns()
    _assert_column(tk, "time (inc)", inclusive, _scaled(inclusive, 10))
    assert tk.inc_metrics == ["time (inc)"]


def test_update_inclusive_columns_union(fill_perfdata):
    """For profiles with different call trees, a node's inclusive value only sums
    the descendants present in that profile. Rows that fill_perfdata filled with
    NaN stay NaN, and rows that are missing are not added."""
    run0 = _make_graphframe(
        [("A", "B"), ("B", "C"), ("A", "E")],
        {"A": 1.0, "B": 2.0, "C": 3.0, "E": 5.0},
    )
    run1 = _make_graphframe(
        [("A", "B"), ("B", "D"), ("A", "E")],
        {"A": 10.0, "B": 20.0, "D": 40.0, "E": 50.0},
    )
    tk = _concat_profiles([run0, run1], fill_perfdata=fill_perfdata)

    tk.update_inclusive_columns()

    inclusive0 = {"A": 11.0, "B": 5.0, "C": 3.0, "E": 5.0}
    inclusive1 = {"A": 120.0, "B": 60.0, "D": 40.0, "E": 50.0}
    if fill_perfdata:
        inclusive0["D"] = np.nan
        inclusive1["C"] = np.nan
    _assert_column(tk, "time (inc)", inclusive0, inclusive1)


def test_update_inclusive_columns_graph():
    """For a graph that is not a tree, each descendant of a node is counted once,
    including descendants of a node that has several parents."""
    # A -> B -> D -> F, A -> C -> D, and a second root R -> S
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "F"), ("R", "S")]
    times = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "F": 5.0, "R": 6.0, "S": 7.0}
    tk = _concat_profiles(
        [_make_graphframe(edges, times), _make_graphframe(edges, _scaled(times, 10))]
    )
    assert len(tk.graph.roots) == 2
    assert len(tk.get_node("D").parents) == 2

    tk.update_inclusive_columns()

    inclusive = {
        "A": 15.0,
        "B": 11.0,
        "C": 12.0,
        "D": 9.0,
        "F": 5.0,
        "R": 13.0,
        "S": 7.0,
    }
    _assert_column(tk, "time (inc)", inclusive, _scaled(inclusive, 10))


def test_update_inclusive_columns_extra_index_level():
    """With an index level after "profile", such as "rank" for per-rank data,
    inclusive values are computed for each profile and rank."""
    edges = [("A", "B"), ("B", "C")]
    times = {
        ("A", 0): 1.0,
        ("B", 0): 2.0,
        ("C", 0): 3.0,
        ("A", 1): 4.0,
        ("B", 1): 5.0,
        ("C", 1): 6.0,
    }
    tk = _concat_profiles(
        [_make_graphframe(edges, times), _make_graphframe(edges, _scaled(times, 10))]
    )
    assert tk.dataframe.index.names == ["node", "profile", "rank"]

    tk.update_inclusive_columns()

    inclusive = {
        ("A", 0): 6.0,
        ("B", 0): 5.0,
        ("C", 0): 3.0,
        ("A", 1): 15.0,
        ("B", 1): 11.0,
        ("C", 1): 6.0,
    }
    _assert_column(tk, "time (inc)", inclusive, _scaled(inclusive, 10))


def test_squash_updates_inclusive_columns():
    """Squash (here, after a query) recomputes inclusive values for each profile
    from the new graph."""
    tk = _concat_profiles(
        [
            _make_graphframe(TREE, TREE_TIMES),
            _make_graphframe(TREE, _scaled(TREE_TIMES, 10)),
        ]
    )
    tk.update_inclusive_columns()

    # Remove B, so that C and D become children of A
    query = Query().match(
        ".", lambda row: row["name"].apply(lambda name: name != "B").all()
    )
    squashed = tk.query(query)

    inclusive = {"A": 13.0, "C": 3.0, "D": 4.0, "E": 5.0}
    _assert_column(squashed, "time (inc)", inclusive, _scaled(inclusive, 10))


def test_update_inclusive_columns_caliper(
    rajaperf_unique_tunings, intersection, fill_perfdata
):
    """Three profiles whose call trees differ (two CUDA tunings and one
    sequential variant)."""
    tk = Thicket.from_caliperreader(
        rajaperf_unique_tunings,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    _check_against_reference(tk)


def test_update_inclusive_columns_column_multiindex(thicket_axis_columns):
    """After concat_thickets(axis="columns"), metrics are (header, metric) tuples
    and each header's inclusive values are computed separately."""
    _, _, tk = thicket_axis_columns
    assert isinstance(tk.dataframe.columns, pd.MultiIndex)
    _check_against_reference(tk)


def test_update_inclusive_columns_timeseries(example_timeseries):
    """A timeseries Thicket is indexed by ("node", "iteration")."""
    tk = Thicket.from_timeseries(example_timeseries)
    assert tk.dataframe.index.names == ["node", "iteration"]
    _check_against_reference(tk)
