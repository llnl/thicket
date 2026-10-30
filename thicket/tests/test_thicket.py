# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re

import pytest
import numpy as np
import pandas as pd
import hatchet as ht

from thicket import Thicket
import thicket.helpers as helpers


def test_invalid_constructor():
    with pytest.raises(ValueError):
        Thicket(None, None)


def test_resolve_missing_indicies():
    names_0 = ["node", "profile", "rank"]
    names_1 = ["node", "profile"]
    node_0 = ht.node.Node(ht.frame.Frame({"name": "foo", "type": "function"}), hnid=0)
    node_1 = ht.node.Node(ht.frame.Frame({"name": "bar", "type": "function"}), hnid=1)
    df_0 = pd.DataFrame(
        data={"time": np.random.randn(4), "name": ["foo", "foo", "bar", "bar"]},
        index=pd.MultiIndex.from_product(
            [[node_0, node_1], ["A"], ["0", "1"]], names=names_0
        ),
    )
    df_1 = pd.DataFrame(
        data={"time": np.random.randn(2), "name": ["foo", "bar"]},
        index=pd.MultiIndex.from_product([[node_0, node_1], ["B"]], names=names_1),
    )
    t_graph = ht.graph.Graph([])
    th_0 = Thicket(graph=t_graph, dataframe=df_0)
    th_1 = Thicket(graph=t_graph, dataframe=df_1)

    helpers._resolve_missing_indicies([th_0, th_1])

    assert th_0.dataframe.index.names == th_1.dataframe.index.names
    assert set(names_0).issubset(th_0.dataframe.index.names)
    assert set(names_0).issubset(th_1.dataframe.index.names)
    assert set(names_1).issubset(th_0.dataframe.index.names)
    assert set(names_1).issubset(th_1.dataframe.index.names)


def test_statsframe(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    def _test_multiindex():
        """Test statsframe when headers are multiindexed."""
        th1 = Thicket.from_caliperreader(rajaperf_seq_O3_1M_cali[0], disable_tqdm=True)
        th2 = Thicket.from_caliperreader(rajaperf_seq_O3_1M_cali[1], disable_tqdm=True)
        th_cj = Thicket.concat_thickets([th1, th2], axis="columns", disable_tqdm=True)

        # Check column format
        assert ("name", "") in th_cj.statsframe.dataframe.columns

    _test_multiindex()

    th = Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali[-1],
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    # Arbitrary value insertion in aggregated statistics table.
    th.statsframe.dataframe["test"] = 1

    # Check that the aggregated statistics table is a Hatchet GraphFrame.
    assert isinstance(th.statsframe, ht.GraphFrame)
    # Check that 'name' column is in dataframe. If not, tree() will not work.
    assert "name" in th.statsframe.dataframe
    # Check length of graph is the same as the dataframe.
    assert len(th.statsframe.graph) == len(th.statsframe.dataframe)

    # Expected tree output
    tree_output = th.statsframe.tree(metric_column="test")

    # Check if tree output is correct.
    assert bool(re.search("1.000.*Algorithm_MEMCPY", tree_output))
    assert bool(re.search("1.000.*Apps_CONVECTION3DPA", tree_output))
    assert bool(re.search("1.000.*Basic_COPY8", tree_output))


def test_metadata_columns_to_perfdata(
    rajaperf_cuda_block128_1M_cali, rajaperf_seq_O3_1M_cali, intersection, fill_perfdata
):
    tk = Thicket.from_caliperreader(
        [rajaperf_cuda_block128_1M_cali[0], rajaperf_seq_O3_1M_cali[0]],
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    tkc1 = tk.deepcopy()

    tk.metadata_columns_to_perfdata(["variant", "tuning"])

    # Check columns added
    assert "variant" in tk.dataframe.columns and "tuning" in tk.dataframe.columns

    # Check overwrite warning raised
    with pytest.warns(UserWarning, match=r"Column .* already exists"):
        tk.metadata_columns_to_perfdata(["variant", "tuning"])

    # Check drop works
    tkc2 = tk.deepcopy()
    tkc2.metadata_columns_to_perfdata("variant", overwrite=True, drop=True)
    assert "variant" not in tkc2.metadata

    # Check error raise for join_key
    tkc2.dataframe = tkc2.dataframe.reset_index(level=tkc2.profile_idx_name, drop=True)
    with pytest.raises(KeyError, match="'profile' must be present"):
        tkc2.metadata_columns_to_perfdata("tuning", overwrite=True)

    # Check alternate join key
    tk.metadata_columns_to_perfdata("ProblemSizeRunParam")
    tk.metadata_columns_to_perfdata("user", join_key="ProblemSizeRunParam")
    assert "user" in tk.dataframe

    # Check column axis Thicket
    # 1. without metadata_key
    gb = tkc1.groupby(["variant", "tuning"])
    ctk = Thicket.concat_thickets(
        thickets=list(gb.values()),
        axis="columns",
        headers=list(gb.keys()),
    )
    ctk.metadata_columns_to_perfdata(
        metadata_columns=[(("Base_CUDA", "block_128"), "ProblemSizeRunParam")]
    )
    assert (("Base_CUDA", "block_128"), "ProblemSizeRunParam") in ctk.dataframe.columns
    # 2. with metadata_key
    ctk2 = Thicket.concat_thickets(
        thickets=list(gb.values()),
        axis="columns",
        headers=list(gb.keys()),
        metadata_key="ProblemSizeRunParam",
    )
    ctk2.metadata_columns_to_perfdata(
        metadata_columns=[(("Base_CUDA", "block_128"), "user")],
        join_key="ProblemSizeRunParam",
    )
    assert (("Base_CUDA", "block_128"), "user") in ctk2.dataframe.columns


def test_perfdata_column_to_statsframe(literal_thickets, mpi_scaling_cali):
    th_single = literal_thickets[1].deepcopy()

    with pytest.raises(KeyError):
        th_single.move_metrics_to_statsframe(["dummy"])

    th_single.move_metrics_to_statsframe(["time"])
    assert all(
        th_single.dataframe["time"].values
        == th_single.statsframe.dataframe["time"].values
    )

    with pytest.raises(KeyError):
        th_single.move_metrics_to_statsframe(["time"])

    th_single.move_metrics_to_statsframe(["time", "memory"], override=True)
    assert all(
        th_single.dataframe["time"].values
        == th_single.statsframe.dataframe["time"].values
    )
    assert all(
        th_single.dataframe["memory"].values
        == th_single.statsframe.dataframe["memory"].values
    )

    th_mpi = Thicket.from_caliperreader(mpi_scaling_cali)
    metrics = ["Min time/rank", "Max time/rank", "Avg time/rank", "Total time"]
    idx = pd.IndexSlice

    with pytest.raises(ValueError):
        th_mpi.move_metrics_to_statsframe(metrics, profile="fake")

    th_mpi.move_metrics_to_statsframe(metrics, profile=th_mpi.profile[0])
    for met in metrics:
        assert all(
            th_mpi.dataframe.loc[idx[:, th_mpi.profile[0]], :][met].values
            == th_mpi.statsframe.dataframe[met].values
        )


def test_thicketize_graphframe(rajaperf_seq_O3_1M_cali):
    ht1 = ht.GraphFrame.from_caliperreader(rajaperf_seq_O3_1M_cali[-1])
    th1 = Thicket.thicketize_graphframe(ht1, rajaperf_seq_O3_1M_cali[-1])

    # Check object types
    assert isinstance(ht1, ht.GraphFrame)
    assert isinstance(th1, Thicket)

    # Check graphs are equivalent
    assert ht1.graph == th1.graph

    # Check dataframes are equivalent when profile level is dropped
    th1_df = th1.dataframe.reset_index(level=th1.profile_idx_name).drop(
        th1.profile_idx_name, axis=1
    )
    assert ht1.dataframe.equals(th1_df)


def test_unique_metadata_base_cuda(
    rajaperf_cuda_block128_1M_cali, intersection, fill_perfdata
):
    t_ens = Thicket.from_caliperreader(
        rajaperf_cuda_block128_1M_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    res = t_ens.get_unique_metadata()
    assert res["systype_build"] == ["blueos_3_ppc64le_ib_p9"]
    assert res["variant"] == ["Base_CUDA"]
    assert res["tuning"] == ["block_128"]


def test_different_profile_idx_name():
    th = Thicket(
        graph=ht.graph.Graph(roots=[]),
        dataframe=pd.DataFrame(
            index=pd.MultiIndex(
                names=["node", "profile2"], levels=[[], []], codes=[[], []]
            )
        ),
        profile_idx_name="profile2",
    )
    assert th.profile_idx_name == "profile2"


def test_replace_profile_index(mpi_scaling_cali, intersection, fill_perfdata):
    th = Thicket.from_caliperreader(
        mpi_scaling_cali[-1],
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert th.profile_idx_name == "profile"
    assert th.dataframe.index.names[1] == "profile"
    assert th.metadata.index.name == "profile"

    th = th.replace_profile_index(
        metadata_key="mpi.world.size",
    )

    assert th.profile_idx_name == "mpi.world.size"
    assert th.dataframe.index.names[1] == "mpi.world.size"
    assert th.metadata.index.name == "mpi.world.size"

    th2 = Thicket.from_caliperreader(
        mpi_scaling_cali,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    th2 = th2.replace_profile_index(
        metadata_key="mpi.world.size",
    )

    # Check new profile
    assert sorted(th2.profile) == [27, 64, 125, 216, 343]
    # check "343" mapped correctly to ".../343_cores.cali"
    for ranks, file in th2.profile_mapping.items():
        assert str(ranks) in file
    # Check metadata
    assert int(th2.metadata.loc[27, "elapsed_time"]) == 47
    assert int(th2.metadata.loc[64, "elapsed_time"]) == 55
    assert int(th2.metadata.loc[125, "elapsed_time"]) == 56
    assert int(th2.metadata.loc[216, "elapsed_time"]) == 42
    assert int(th2.metadata.loc[343, "elapsed_time"]) == 52
    # Check PerfData
    assert int(th2.dataframe.loc[(th2.get_node("main"), 27), "Avg time/rank"]) == 47
    assert int(th2.dataframe.loc[(th2.get_node("main"), 64), "Avg time/rank"]) == 55
    assert int(th2.dataframe.loc[(th2.get_node("main"), 125), "Avg time/rank"]) == 56
    assert int(th2.dataframe.loc[(th2.get_node("main"), 216), "Avg time/rank"]) == 42
    assert int(th2.dataframe.loc[(th2.get_node("main"), 343), "Avg time/rank"]) == 52
