# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op
from .ttest import __ttest

__statistical_tests = {"ttest": __ttest}


@cache_stats_op
def preference(thicket, columns, comparison_func, *args, test="ttest", **kwargs):
    """Determine a preference between compilers, architecture, platform, etc.

    Designed to take in a thicket and will append eight total columns to the
    aggregated statistics table. As a note, preferred will stand for the preferred
    choice between two options.
        - <columns[0]>_mean
        - <columns[0]>_std
        - <columns[1]>_mean
        - <columns[1]>_std
        - <columns[0]> vs <columns[1]>_tvalue
        - <columns[0]> vs <columns[1]>_tstatistic
        - <columns[0]> vs <columns[1]>_std_preferred
        - <columns[0]> vs <columns[1]>_mean_preferred

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to determine a preference for.
            Note, if using a columnar joined thicket a list of tuples must be passed in
            with the format (column index, column name). List should be length 2.
        comparison_func (function): User-defined python or lambda function to decide a
            preference.
        test (str): User-selected test.

    Returns:
        (list): returns a list of output statsframe column names
    """
    if len(columns) != 2:
        raise ValueError("Must specify 2 columns in columns=.")

    if test not in __statistical_tests.keys():
        raise ValueError("Test is not available.")

    verify_thicket_structures(thicket.dataframe, index=["node"], columns=columns)

    output_column_names = []

    if test == "ttest":
        tvalue, t_statistics = __statistical_tests[test](
            thicket, columns, *args, **kwargs
        )

    pref_mean = []
    pref_std = []

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        for i, t_statistic in enumerate(t_statistics):
            if t_statistic < -1 * tvalue or t_statistic > tvalue:
                pref_mean.append(
                    comparison_func(
                        thicket.statsframe.dataframe[columns[0] + "_mean"].iloc[i],
                        thicket.statsframe.dataframe[columns[1] + "_mean"].iloc[i],
                    )
                )
                pref_std.append(
                    comparison_func(
                        thicket.statsframe.dataframe[columns[0] + "_std"].iloc[i],
                        thicket.statsframe.dataframe[columns[1] + "_std"].iloc[i],
                    )
                )
            else:
                pref_mean.append("No preference")
                pref_std.append("No preference")
        aggregated_cols = columns[0] + " vs " + columns[1]
        thicket.statsframe.dataframe[aggregated_cols + "_std_preferred"] = pref_std
        thicket.statsframe.dataframe[aggregated_cols + "_mean_preferred"] = pref_mean

        output_column_names.append(aggregated_cols + "_std_preferred")
        output_column_names.append(aggregated_cols + "_mean_preferred")
    # columnar joined thicket object
    else:
        idx_mean = [(index, col + "_mean") for index, col in columns]
        idx_std = [(index, col + "_std") for index, col in columns]
        for i, t_statistic in enumerate(t_statistics):
            if t_statistic < -1 * tvalue or t_statistic > tvalue:
                pref_mean.append(
                    comparison_func(
                        thicket.statsframe.dataframe[idx_mean[0]].iloc[i],
                        thicket.statsframe.dataframe[idx_mean[1]].iloc[i],
                    )
                )
                pref_std.append(
                    comparison_func(
                        thicket.statsframe.dataframe[idx_std[0]].iloc[i],
                        thicket.statsframe.dataframe[idx_std[1]].iloc[i],
                    )
                )
            else:
                pref_mean.append("No preference")
                pref_std.append("No preference")

        aggregated_cols = (
            str(columns[0]).replace("'", "") + " vs " + str(columns[1]).replace("'", "")
        )

        col_name = ["Preference", aggregated_cols]
        thicket.statsframe.dataframe[
            (col_name[0], col_name[1] + "_std_preferred")
        ] = pref_std
        thicket.statsframe.dataframe[
            (col_name[0], col_name[1] + "_mean_preferred")
        ] = pref_mean

        output_column_names.append((col_name[0], col_name[1] + "_std_preferred"))
        output_column_names.append((col_name[0], col_name[1] + "_mean_preferred"))

    return output_column_names
