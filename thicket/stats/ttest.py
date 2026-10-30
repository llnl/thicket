# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
from scipy.stats import ttest_ind_from_stats
from scipy.stats import t

import thicket as th


def __ttest(thicket, columns, alpha=0.05, *args, **kwargs):
    """Perform a ttest on a user-selected thicket and columns.

    Designed to take in a thicket and two columns. For this private function a tvalue
    and list of tstatistics will be returned to preference.py.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to determine a preference for.
            Note, if using a columnar joined thicket a list of tuples must be passed in
            with the format (column index, column name).
        alpha (double): Threshold for statistical significance. Value must be between 0
            and 1. Default is 0.05.

    Returns:
       tvalue (double): Value to be used to determine a preference within preference.py.
       tstatistics (list): List of values to be used to determine a preference within
            preference.py.
    """
    # check to see if alpha value is between 0 and 1
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Value for alpha argument must be between 0 and 1.")

    # check columns contain two columns
    if len(columns) != 2:
        raise ValueError("Columns must be a list of length 2.")

    n = pd.unique(thicket.dataframe.reset_index()["node"])[0]

    # nobs for parameter one for ttest
    nobs_column1 = len(thicket.dataframe.loc[n][columns[0]])
    # nobs for parameter two for ttest
    nobs_column2 = len(thicket.dataframe.loc[n][columns[0]])

    # subtract by len(columns) due to estimating a t-test with two parameters
    # alpha/2 is done for a two tail t-test
    tvalue = t.ppf(q=1 - alpha / 2, df=nobs_column1 + nobs_column2 - len(columns))

    th.stats.mean(thicket, columns)
    th.stats.std(thicket, columns)

    # thicket object with columnar index
    if thicket.dataframe.columns.nlevels > 1:
        mean_columns = [(idx, col + "_mean") for idx, col in columns]
        std_columns = [(idx, col + "_std") for idx, col in columns]
        t_statistics = []
        for i in range(0, len(thicket.statsframe.dataframe)):
            tStatistic = ttest_ind_from_stats(
                mean1=thicket.statsframe.dataframe[mean_columns[0]].iloc[i],
                std1=thicket.statsframe.dataframe[std_columns[0]].iloc[i],
                nobs1=nobs_column1,
                mean2=thicket.statsframe.dataframe[mean_columns[1]].iloc[i],
                std2=thicket.statsframe.dataframe[std_columns[1]].iloc[i],
                nobs2=nobs_column2,
                equal_var=False,
            )

            t_statistics.append(tStatistic.statistic)

        # store results into thicket's aggregated statistics table
        aggregated_cols = (
            str(columns[0]).replace("'", "") + " vs " + str(columns[1]).replace("'", "")
        )

        thicket.statsframe.dataframe[
            (
                "Preference",
                aggregated_cols + "_tvalue",
            )
        ] = tvalue
        thicket.statsframe.dataframe[
            (
                "Preference",
                aggregated_cols + "_tstatistic",
            )
        ] = t_statistics

        return tvalue, t_statistics
    # thicket object without columnar index
    else:
        # gather mean and std columns
        mean_columns = [col + "_mean" for col in columns]
        std_columns = [col + "_std" for col in columns]
        t_statistics = []
        for i in range(0, len(thicket.statsframe.dataframe)):
            tStatistic = ttest_ind_from_stats(
                mean1=thicket.statsframe.dataframe[mean_columns[0]].iloc[i],
                std1=thicket.statsframe.dataframe[std_columns[0]].iloc[i],
                nobs1=nobs_column1,
                mean2=thicket.statsframe.dataframe[mean_columns[1]].iloc[i],
                std2=thicket.statsframe.dataframe[std_columns[1]].iloc[i],
                nobs2=nobs_column2,
                equal_var=False,
            )

            t_statistics.append(tStatistic.statistic)

        # store results into thicket's aggregated statistics table
        aggregated_cols = columns[0] + " vs " + columns[1]

        thicket.statsframe.dataframe[aggregated_cols + "_tvalue"] = tvalue
        thicket.statsframe.dataframe[aggregated_cols + "_tstatistic"] = t_statistics

        return tvalue, t_statistics
