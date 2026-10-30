# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
from scipy import stats

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op


@cache_stats_op
def check_normality(thicket, columns=None):
    """Determine if the data is normal or non-normal for each node in the performance
    data table.

    Designed to take in a thicket, and append one or more columns to the aggregated
    statistics table. A true boolean value will be appended if the data is normal and a
    false boolean value will be appended if the data is non-normal.

    For this test, the more data the better. Preferably you would want to have 20 data
    points (20 files) in a dataset to have an accurate result.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to perform normality test on.
            Note, if using a columnar joined thicket a list of tuples must be passed in
            with the format (column index, column name).

    Returns:
        (list): returns a list of output statsframe column names
    """
    if columns is None:
        raise ValueError(
            "To see a list of valid columns, run 'Thicket.performance_cols'."
        )

    verify_thicket_structures(thicket.dataframe, index=["node"], columns=columns)

    output_column_names = []

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        df = (
            thicket.dataframe.select_dtypes(include="number")
            .reset_index()
            .groupby("node")
            .agg(stats.shapiro)
        )
        for column in columns:
            output_column_names.append(column + "_normality")
            for i in range(0, len(df[column])):
                pvalue = df[column].iloc[i].pvalue

                if pvalue < 0.05:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], column + "_normality"
                    ] = "False"
                elif pvalue > 0.05:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], column + "_normality"
                    ] = "True"
                else:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], column + "_normality"
                    ] = pd.NA
                # check to see if exclusive metric
                if column in thicket.exc_metrics:
                    thicket.statsframe.exc_metrics.append(column + "_normality")
                # check to see if inclusive metric
                else:
                    thicket.statsframe.inc_metrics.append(column + "_normality")
    # columnar joined thicket object
    else:
        df = (
            thicket.dataframe.select_dtypes(include="number")
            .reset_index(level=1)
            .groupby("node")
            .agg(stats.shapiro)
        )
        for idx, column in columns:
            output_column_names.append((idx, column + "_normality"))
            for i in range(0, len(df[(idx, column)])):
                pvalue = df[(idx, column)].iloc[i].pvalue

                if pvalue < 0.05:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], (idx, column + "_normality")
                    ] = "False"
                elif pvalue > 0.05:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], (idx, column + "_normality")
                    ] = "True"
                else:
                    thicket.statsframe.dataframe.loc[
                        df.index[i], (idx, column + "_normality")
                    ] = pd.NA
            # check to see if exclusive metric
            if (idx, column) in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append((idx, column + "_normality"))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append((idx, column + "_normality"))

        # sort columns in index
        thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
