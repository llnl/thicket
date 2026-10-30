# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import math
import warnings

import numpy as np
import pytest

import thicket as th


def test_mean(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.mean(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_mean" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_mean"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_mean" in th_ens.statsframe.show_metric_columns()


def test_mean_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.mean(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_mean") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_mean") in combined_th.statsframe.show_metric_columns()


def test_sum(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.sum(th_ens, columns=["Min time/rank"])

    print(th_ens.dataframe["Min time/rank"])

    print(
        th_ens.statsframe.dataframe.loc[
            th_ens.get_node("RAJAPerf"), "Min time/rank_sum"
        ]
    )

    assert (
        abs(
            th_ens.statsframe.dataframe.loc[
                th_ens.get_node("RAJAPerf"), "Min time/rank_sum"
            ]
            - 399.65
        )
        < 1e-2
    )

    assert "Min time/rank_sum" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_sum"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_sum" in th_ens.statsframe.show_metric_columns()


def test_sum_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.sum(combined_th, columns=[(idx, "Min time/rank")])

    assert (
        abs(
            combined_th.statsframe.dataframe.loc[
                combined_th.get_node("RAJAPerf"), (idx, "Min time/rank_sum")
            ]
            - 5.16
        )
        < 1e-2
    )

    assert (idx, "Min time/rank_sum") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_sum",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_sum") in combined_th.statsframe.show_metric_columns()


def test_median(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.median(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_median" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_median"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_median" in th_ens.statsframe.show_metric_columns()


def test_median_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.median(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_median") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_median",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_median") in combined_th.statsframe.show_metric_columns()


def test_minimum(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.minimum(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_min" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_min"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_min" in th_ens.statsframe.show_metric_columns()


def test_minimum_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.minimum(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_min") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_min",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_min") in combined_th.statsframe.show_metric_columns()


def test_maximum(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.maximum(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_max" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_max"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_max" in th_ens.statsframe.show_metric_columns()


def test_maximum_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.maximum(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_max") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_max",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_max") in combined_th.statsframe.show_metric_columns()


def test_std(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.std(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_std" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_std"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_std" in th_ens.statsframe.show_metric_columns()


def test_std_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.std(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_std") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_std") in combined_th.statsframe.show_metric_columns()


def test_aggregation_values(
    rajaperf_cuda_block128_1M_cali, rajaperf_seq_O3_1M_cali, intersection, fill_perfdata
):
    """Check the values of the aggregation functions against NumPy.

    Missing values (e.g., rows added by fill_perfdata) are skipped, and std and
    variance are sample statistics (ddof=1).
    """
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_cuda_block128_1M_cali[:3] + rajaperf_seq_O3_1M_cali[:3],
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )
    column = "Min time/rank"
    stats_funcs = {
        "mean": (th.stats.mean, np.nanmean),
        "median": (th.stats.median, np.nanmedian),
        "sum": (th.stats.sum, np.nansum),
        "max": (th.stats.maximum, np.nanmax),
        "min": (th.stats.minimum, np.nanmin),
        "std": (th.stats.std, lambda x: np.nanstd(x, ddof=1)),
        "var": (th.stats.variance, lambda x: np.nanvar(x, ddof=1)),
    }
    for stats_func, _ in stats_funcs.values():
        stats_func(th_ens, columns=[column])

    for node in th_ens.statsframe.dataframe.index:
        values = th_ens.dataframe.loc[node, column].to_numpy()
        for suffix, (_, numpy_func) in stats_funcs.items():
            with warnings.catch_warnings():
                # e.g., "Degrees of freedom <= 0" for single values
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = numpy_func(values)
            assert th_ens.statsframe.dataframe.loc[
                node, f"{column}_{suffix}"
            ] == pytest.approx(expected, nan_ok=True)


def test_percentiles(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )
    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.percentiles(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_percentiles_25" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_percentiles_50" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_percentiles_75" in th_ens.statsframe.dataframe.columns

    assert (
        "Min time/rank_percentiles_25"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_percentiles_50"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_percentiles_75"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_percentiles_25" in th_ens.statsframe.show_metric_columns()
    assert "Min time/rank_percentiles_50" in th_ens.statsframe.show_metric_columns()
    assert "Min time/rank_percentiles_75" in th_ens.statsframe.show_metric_columns()


def test_percentiles_none(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    th.stats.percentiles(th_ens, columns=["Min time/rank"], percentiles=None)

    assert "Min time/rank_percentiles_25" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_percentiles_50" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_percentiles_75" in th_ens.statsframe.dataframe.columns


def test_percentiles_single_value(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    th.stats.percentiles(th_ens, columns=["Min time/rank"], percentiles=[0.3])

    assert "Min time/rank_percentiles_30" in th_ens.statsframe.dataframe.columns

    assert (
        "Min time/rank_percentiles_30"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )

    assert "Min time/rank_percentiles_30" in th_ens.statsframe.show_metric_columns()


def test_percentiles_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.percentiles(combined_th, columns=[(idx, "Min time/rank")])

    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.show_metric_columns()

    th.stats.percentiles(
        combined_th, columns=[(idx, "Min time/rank")], percentiles=[0.4]
    )

    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx,
        "Min time/rank_percentiles_25",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_percentiles_50",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_percentiles_75",
    ) in combined_th.statsframe.show_metric_columns()

    th.stats.percentiles(
        combined_th, columns=[(idx, "Min time/rank")], percentiles=[0.4]
    )

    assert (
        idx,
        "Min time/rank_percentiles_40",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_percentiles_40",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_percentiles_40",
    ) in combined_th.statsframe.show_metric_columns()


def test_variance(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.variance(th_ens, columns=["Min time/rank"])

    assert "Min time/rank_var" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_var"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_var" in th_ens.statsframe.show_metric_columns()


def test_variance_columnar_join(thicket_axis_columns, intersection, fill_perfdata):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.variance(combined_th, columns=[(idx, "Min time/rank")])

    assert (idx, "Min time/rank_var") in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_var",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (idx, "Min time/rank_var") in combined_th.statsframe.show_metric_columns()


def test_normality(rajaperf_cuda_block128_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_cuda_block128_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.check_normality(th_ens, columns=["Min time/rank"])

    assert th_ens.statsframe.dataframe["Min time/rank_normality"].iloc[0] in {
        "True",
        "False",
    }

    assert "Min time/rank_normality" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_normality"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert "Min time/rank_normality" in th_ens.statsframe.show_metric_columns()


def test_normality_columnar_join(thicket_axis_columns, stats_thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    sthicket_list, sthicket_list_cp, scombined_th = stats_thicket_axis_columns
    # new data must be added before uncommenting, need 3 or more datapoints
    # idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    for idx in ["Cuda 1", "Cuda 2"]:
        th.stats.check_normality(scombined_th, columns=[(idx, "Min time/rank")])

        assert (
            idx,
            "Min time/rank_normality",
        ) in scombined_th.statsframe.dataframe.columns
        assert (
            (
                idx,
                "Min time/rank_normality",
            )
            in scombined_th.statsframe.exc_metrics + scombined_th.statsframe.inc_metrics
        )
        assert (
            idx,
            "Min time/rank_normality",
        ) in scombined_th.statsframe.show_metric_columns()


def test_correlation(rajaperf_cuda_block128_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_cuda_block128_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.correlation_nodewise(
        th_ens, column1="Min time/rank", column2="Max time/rank", correlation="pearson"
    )

    assert (
        "Min time/rank_vs_Max time/rank pearson" in th_ens.statsframe.dataframe.columns
    )


def test_correlation_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.correlation_nodewise(
        combined_th,
        column1=(idx[0], "Min time/rank"),
        column2=(idx[1], "Max time/rank"),
        correlation="spearman",
    )

    assert (
        "Union statistics",
        "Min time/rank_vs_Max time/rank spearman",
    ) in combined_th.statsframe.dataframe.columns


def test_boxplot(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_ens = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=True,
    )

    assert sorted(th_ens.dataframe.index.get_level_values(0).unique()) == sorted(
        th_ens.statsframe.dataframe.index.values
    )

    assert list(th_ens.statsframe.dataframe.columns) == ["name"]

    th.stats.calc_boxplot_statistics(
        th_ens, columns=["Min time/rank"], quartiles=[0.25, 0.50, 0.75]
    )

    assert "Min time/rank_q1(0.25, 0.5, 0.75)" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_q2(0.25, 0.5, 0.75)" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_q3(0.25, 0.5, 0.75)" in th_ens.statsframe.dataframe.columns
    assert "Min time/rank_iqr(0.25, 0.5, 0.75)" in th_ens.statsframe.dataframe.columns
    assert (
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.dataframe.columns
    )
    assert (
        "Min time/rank_upperfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.dataframe.columns
    )
    assert (
        "Min time/rank_outliers(0.25, 0.5, 0.75)" in th_ens.statsframe.dataframe.columns
    )

    assert (
        "Min time/rank_q1(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_q2(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_q3(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_iqr(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_upperfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )
    assert (
        "Min time/rank_outliers(0.25, 0.5, 0.75)"
        in th_ens.statsframe.exc_metrics + th_ens.statsframe.inc_metrics
    )

    assert (
        "Min time/rank_q1(0.25, 0.5, 0.75)" in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_q2(0.25, 0.5, 0.75)" in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_q3(0.25, 0.5, 0.75)" in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_iqr(0.25, 0.5, 0.75)" in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_upperfence(0.25, 0.5, 0.75)"
        in th_ens.statsframe.show_metric_columns()
    )
    assert (
        "Min time/rank_outliers(0.25, 0.5, 0.75)"
        in th_ens.statsframe.show_metric_columns()
    )


def test_boxplot_columnar_join(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    idx = combined_th.dataframe.columns.levels[0][0]
    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.calc_boxplot_statistics(
        combined_th, columns=[(idx, "Min time/rank")], quartiles=[0.25, 0.50, 0.75]
    )

    assert (
        idx,
        "Min time/rank_q1(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_q2(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_q3(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_iqr(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_upperfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns
    assert (
        idx,
        "Min time/rank_outliers(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.dataframe.columns

    assert (
        idx,
        "Min time/rank_q1(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_q2(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_q3(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_iqr(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_upperfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics
    assert (
        idx,
        "Min time/rank_outliers(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx,
        "Min time/rank_q1(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_q2(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_q3(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_iqr(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_lowerfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_upperfence(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()
    assert (
        idx,
        "Min time/rank_outliers(0.25, 0.5, 0.75)",
    ) in combined_th.statsframe.show_metric_columns()


def test_bhattacharyya_distance(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    combined_th_cpy = th.Thicket.copy(combined_th)

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "bhattacharyya_distance"

    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    stats_out = th.stats.bhattacharyya_distance(
        combined_th, columns=columns, output_column_name=output_column_name
    )

    assert (
        output_column_name,
        "",
    ) in combined_th.statsframe.dataframe.columns.to_list()

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[1],
        "Min time/rank_std",
    ) in combined_th.statsframe.show_metric_columns()

    # Compare scoring values
    num_nodes = len(combined_th_cpy.dataframe.index.get_level_values(0).unique())

    mean_columns = th.stats.mean(combined_th_cpy, columns)
    std_columns = th.stats.std(combined_th_cpy, columns)

    means_target1 = combined_th_cpy.statsframe.dataframe[mean_columns[0]].to_numpy()
    means_target2 = combined_th_cpy.statsframe.dataframe[mean_columns[1]].to_numpy()

    stds_target1 = combined_th_cpy.statsframe.dataframe[std_columns[0]].to_numpy()
    stds_target2 = combined_th_cpy.statsframe.dataframe[std_columns[1]].to_numpy()

    results = []

    for i in range(num_nodes):
        try:
            result = 0.25 * np.log(
                0.25
                * (
                    (stds_target1[i] ** 2 / stds_target2[i] ** 2)
                    + (stds_target2[i] ** 2 / stds_target1[i] ** 2)
                    + 2
                )
            ) + 0.25 * (
                (means_target1[i] - means_target2[i]) ** 2
                / (stds_target1[i] ** 2 + stds_target2[i] ** 2)
            )
        except ZeroDivisionError:
            result = np.nan

        results.append(result)

    results = [-1 if math.isnan(x) else x for x in results]

    distance = combined_th.statsframe.dataframe[stats_out[0]].to_list()
    distance = [-1 if math.isnan(x) else x for x in distance]

    assert results == distance


def test_hellinger_distance(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    combined_th_cpy = th.Thicket.copy(combined_th)

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "hellinger_distance"

    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    stats_out = th.stats.hellinger_distance(
        combined_th, columns=columns, output_column_name=output_column_name
    )

    assert (
        output_column_name,
        "",
    ) in combined_th.statsframe.dataframe.columns.to_list()

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (idx[1], "Min time/rank_std") in combined_th.statsframe.show_metric_columns()

    # Compare scoring values
    num_nodes = len(combined_th_cpy.dataframe.index.get_level_values(0).unique())

    mean_columns = th.stats.mean(combined_th_cpy, columns)
    std_columns = th.stats.std(combined_th_cpy, columns)

    means_target1 = combined_th_cpy.statsframe.dataframe[mean_columns[0]].to_numpy()
    means_target2 = combined_th_cpy.statsframe.dataframe[mean_columns[1]].to_numpy()

    stds_target1 = combined_th_cpy.statsframe.dataframe[std_columns[0]].to_numpy()
    stds_target2 = combined_th_cpy.statsframe.dataframe[std_columns[1]].to_numpy()

    results = []

    for i in range(num_nodes):
        try:
            result = 1 - math.sqrt(
                (2 * stds_target1[i] * stds_target2[i])
                / (stds_target1[i] ** 2 + stds_target2[i] ** 2)
            ) * math.exp(
                -0.25
                * ((means_target1[i] - means_target2[i]) ** 2)
                / (stds_target1[i] ** 2 + stds_target2[i] ** 2)
            )
        except ZeroDivisionError:
            result = np.nan

        results.append(result)

    results = [-1 if math.isnan(x) else x for x in results]

    distance = combined_th.statsframe.dataframe[stats_out[0]].to_list()
    distance = [-1 if math.isnan(x) else x for x in distance]

    assert results == distance


def test_score_delta_mean_delta_stdnorm(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    combined_th_cpy = th.Thicket.copy(combined_th)

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "score_delta_mean_delta_stdnorm"

    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    output_column_names = th.stats.score_delta_mean_delta_stdnorm(
        combined_th,
        columns=columns,
        output_column_name=output_column_name,
    )

    assert len(output_column_names) == 1

    assert (
        "Scoring",
        "score_delta_mean_delta_stdnorm",
    ) in combined_th.statsframe.dataframe.columns

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (idx[0], "Min time/rank_std") in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (idx[1], "Min time/rank_std") in combined_th.statsframe.show_metric_columns()

    # Compare scoring values
    num_nodes = len(combined_th_cpy.dataframe.index.get_level_values(0).unique())

    th.stats.mean(combined_th_cpy, columns)
    th.stats.std(combined_th_cpy, columns)

    means_target1 = combined_th_cpy.statsframe.dataframe[
        (columns[0][0], "{}_mean".format(columns[0][1]))
    ].to_list()
    means_target2 = combined_th_cpy.statsframe.dataframe[
        (columns[1][0], "{}_mean".format(columns[1][1]))
    ].to_list()

    stds_target1 = combined_th_cpy.statsframe.dataframe[
        (columns[0][0], "{}_std".format(columns[0][1]))
    ].to_list()
    stds_target2 = combined_th_cpy.statsframe.dataframe[
        (columns[1][0], "{}_std".format(columns[1][1]))
    ].to_list()

    results = []

    for i in range(num_nodes):
        result = (means_target1[i] - means_target2[i]) + (
            (stds_target1[i] - stds_target2[i])
            / (np.abs(means_target1[i] - means_target2[i]))
        )
        results.append(result)

    results = [-1 if math.isnan(x) else x for x in results]

    th_score = combined_th.statsframe.dataframe[
        ("Scoring", output_column_name)
    ].to_list()
    th_score = [-1 if math.isnan(x) else x for x in th_score]

    assert results == th_score


def test_score_delta_mean_delta_coefficient_of_variation(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns
    combined_th_cpy = th.Thicket.copy(combined_th)

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "score_delta_mean_delta_coefficient_of_variation"

    assert sorted(combined_th.dataframe.index.get_level_values(0).unique()) == sorted(
        combined_th.statsframe.dataframe.index.values
    )

    assert list(combined_th.statsframe.dataframe.columns) == [("name", "")]

    th.stats.score_delta_mean_delta_coefficient_of_variation(
        combined_th,
        columns=columns,
        output_column_name=output_column_name,
    )

    assert ("Scoring", output_column_name) in combined_th.statsframe.dataframe.columns

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[1],
        "Min time/rank_mean",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (
        idx[0],
        "Min time/rank_std",
    ) in combined_th.statsframe.show_metric_columns()

    assert (
        idx[1],
        "Min time/rank_std",
    ) in combined_th.statsframe.exc_metrics + combined_th.statsframe.inc_metrics

    assert (idx[1], "Min time/rank_std") in combined_th.statsframe.show_metric_columns()

    # Compare scoring values
    num_nodes = len(combined_th_cpy.dataframe.index.get_level_values(0).unique())

    th.stats.mean(combined_th_cpy, columns)
    th.stats.std(combined_th_cpy, columns)

    means_target1 = combined_th_cpy.statsframe.dataframe[
        (columns[0][0], "{}_mean".format(columns[0][1]))
    ].to_list()
    means_target2 = combined_th_cpy.statsframe.dataframe[
        (columns[1][0], "{}_mean".format(columns[1][1]))
    ].to_list()

    stds_target1 = combined_th_cpy.statsframe.dataframe[
        (columns[0][0], "{}_std".format(columns[0][1]))
    ].to_list()
    stds_target2 = combined_th_cpy.statsframe.dataframe[
        (columns[1][0], "{}_std".format(columns[1][1]))
    ].to_list()

    results = []

    for i in range(num_nodes):
        result = (
            (means_target1[i] - means_target2[i])
            + (stds_target1[i] / means_target1[i])
            - (stds_target2[i] / means_target2[i])
        )

        results.append(result)

    results = [-1 if math.isnan(x) else x for x in results]

    th_score = combined_th.statsframe.dataframe[
        ("Scoring", output_column_name)
    ].to_list()
    th_score = [-1 if math.isnan(x) else x for x in th_score]

    assert results == th_score


def test_score_bhattacharyya(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "bhattacharyya_score"

    # Default Case
    combined_th_cpy = th.Thicket.copy(combined_th)
    stats_out = th.stats.score_bhattacharyya(
        combined_th_cpy, columns=columns, output_column_name=output_column_name
    )

    assert (
        "Scoring",
        output_column_name,
    ) in combined_th_cpy.statsframe.dataframe.columns

    combined_th_cpy = th.Thicket.copy(combined_th)

    # Calculate Distance and assign static signage
    distance_col = th.stats.bhattacharyya_distance(
        combined_th_cpy, columns, output_column_name
    )[0]
    distance_data = combined_th_cpy.statsframe.dataframe[distance_col]

    multiplier = distance_data.apply(lambda row: -1)
    distance_data = distance_data * multiplier

    # This comparison function makes sure that all scores are negative
    comparison_func = lambda x, y: True  # noqa: E731

    stats_out = th.stats.score_bhattacharyya(
        combined_th_cpy,
        columns=columns,
        output_column_name=output_column_name,
        comparison_func=comparison_func,
    )

    # NaN, and Inf are not comparable, removed them
    combined_th_cpy.statsframe.dataframe = combined_th_cpy.statsframe.dataframe.replace(
        [np.inf, -np.inf, np.nan], -1
    )
    distance_data = distance_data.replace([np.inf, -np.inf, np.nan], -1)

    assert (
        combined_th_cpy.statsframe.dataframe[stats_out[0]].tolist()
        == distance_data.tolist()
    )


def test_score_hellinger(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]
    output_column_name = "hellinger_score"

    # Default Case
    combined_th_cpy = th.Thicket.copy(combined_th)
    stats_out = th.stats.score_hellinger(
        combined_th_cpy, columns=columns, output_column_name=output_column_name
    )

    assert (
        "Scoring",
        output_column_name,
    ) in combined_th_cpy.statsframe.dataframe.columns

    combined_th_cpy = th.Thicket.copy(combined_th)

    # Calculate Distance and assign static signage
    distance_col = th.stats.hellinger_distance(
        combined_th_cpy, columns, output_column_name
    )[0]
    distance_data = combined_th_cpy.statsframe.dataframe[distance_col]

    multiplier = distance_data.apply(lambda row: -1)
    distance_data = distance_data * multiplier

    # This comparison function makes sure that all scores are negative
    comparison_func = lambda x, y: True  # noqa: E731

    stats_out = th.stats.score_hellinger(
        combined_th_cpy,
        columns=columns,
        output_column_name=output_column_name,
        comparison_func=comparison_func,
    )

    # NaN, and Inf are not comparable, removed them
    combined_th_cpy.statsframe.dataframe = combined_th_cpy.statsframe.dataframe.replace(
        [np.inf, -np.inf, np.nan], -1
    )
    distance_data = distance_data.replace([np.inf, -np.inf, np.nan], -1)

    assert (
        combined_th_cpy.statsframe.dataframe[stats_out[0]].tolist()
        == distance_data.tolist()
    )


def test_reapply_statsframe_operations(
    rajaperf_seq_O3_1M_cali, intersection, fill_perfdata
):
    th_1 = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=False,
    )

    th.stats.mean(th_1, columns=["Min time/rank"])

    assert len(th_1.statsframe_ops_cache) == 1

    statsframe_copy = th_1.statsframe.dataframe.copy()
    th_1.statsframe.dataframe = th.helpers._new_statsframe_df(
        th_1.dataframe, multiindex=False
    )

    th_1.reapply_stats_operations()

    comp_val = statsframe_copy.eq(th_1.statsframe.dataframe)

    assert all([comp_val[c].all() for c in comp_val.columns])


def test_cache_decorator(rajaperf_seq_O3_1M_cali, intersection, fill_perfdata):
    th_1 = th.Thicket.from_caliperreader(
        rajaperf_seq_O3_1M_cali,
        node_ordering=False,
        intersection=intersection,
        fill_perfdata=fill_perfdata,
        disable_tqdm=False,
    )

    th.stats.mean(th_1, columns=["Min time/rank"])

    assert len(th_1.statsframe_ops_cache) == 1
    assert (
        len(th_1.statsframe_ops_cache[list(th_1.statsframe_ops_cache.keys())[0]]) == 1
    )


def test_confidence_interval(thicket_axis_columns):
    thicket_list, thicket_list_cp, combined_th = thicket_axis_columns

    idx = list(combined_th.dataframe.columns.levels[0][0:2])
    columns = [(idx[0], "Min time/rank"), (idx[1], "Min time/rank")]

    with pytest.raises(
        ValueError, match="Value passed to 'columns' must be of type list."
    ):
        th.stats.confidence_interval(combined_th, columns="columns")

    with pytest.raises(
        ValueError,
        match="Value passed to 'confidence_level' must be of type float.",
    ):
        th.stats.confidence_interval(
            combined_th, columns=columns, confidence_level="0.95"
        )

    with pytest.raises(
        ValueError,
        match=r"Value passed to 'confidence_level' must be in the range of \(0, 1\).",
    ):
        th.stats.confidence_interval(combined_th, columns=columns, confidence_level=1.2)

    # Hardcoded cases
    columns = [("block_128", "Avg time/rank"), ("default", "Avg time/rank")]

    th.stats.confidence_interval(combined_th, columns=columns)

    correct_data = {
        ("block_128", "confidence_interval_0.95_Avg time/rank"): [
            (1.0128577358974717, 4.149184264102528),
            (0.0049270306443246845, 0.012615969355675315),
        ],
        ("default", "confidence_interval_0.95_Avg time/rank"): [
            (43.443961386963, 288.581029613037),
            (1.858945913805485, 10.117062086194515),
        ],
    }

    for col in correct_data.keys():
        for idx, val in enumerate(correct_data[col]):
            assert combined_th.statsframe.dataframe[col].iloc[idx] == val
