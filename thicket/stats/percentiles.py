# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op


@cache_stats_op
def percentiles(thicket, columns=None, percentiles=[0.25, 0.50, 0.75]):
    """Calculate the q-th percentile for each node in the performance data table.

    Designed to take in a thicket, and append one or more columns to the aggregated
    statistics table for the q-th percentile calculation for each node. Each percentile
    calculation is a separate column in the statistics table, where the column will
    have the format: {columnName}_percentiles_{percentile}.

    The 25th percentile is the lower quartile, and is the value at which 25% of the
    answers lie below that value.

    The 50th percentile, is the median and half of the values lie below the median and
    half lie above the median.

    The 75th percentile is the upper quartile, and is the value at which 25% of the
    answers lie above that value and 75% of the answers lie below that value.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to perform percentile
            calculation on. Note if using a columnar joined thicket a list of tuples
            must be passed in with the format (column index, column name).
        percentiles (list): List of percentile values that is desired to be calculated
            for each column in columns. If no list is specified, the default values,
            [0.25, 0.50, 0.75] are used for calculations

    Returns:
        (list): returns a list of output statsframe column names
    """
    if not percentiles:
        percentiles = [0.25, 0.50, 0.75]

    # Enforce that percentiles are in range of [0.0, 1.0]
    for percentile in percentiles:
        if percentile < 0.0 or percentile > 1.0:
            raise ValueError(
                "Percentile {} is out of range of [0.0, 1.0]".format(percentile)
            )

    if columns is None:
        raise ValueError(
            "To see a list of valid columns, run 'Thicket.performance_cols'."
        )

    verify_thicket_structures(thicket.dataframe, index=["node"], columns=columns)

    output_column_names = []

    # select numeric columns within thicket (.quantiles) will not work without this step
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        df_num = thicket.dataframe.select_dtypes(include=numerics)[columns]
        df = df_num.reset_index().groupby("node").quantile(percentiles)
        for column in columns:
            calculated_percentiles = []
            for node in pd.unique(df.reset_index()["node"]):
                calculated_percentiles.append(list(df.loc[node][column]))

            for index, percentile in enumerate(percentiles):
                column_to_append = column + "_percentiles_" + str(int(percentile * 100))
                output_column_names.append(column_to_append)
                thicket.statsframe.dataframe[column_to_append] = [
                    x[index] for x in calculated_percentiles
                ]

                # check to see if exclusive metric and that the metric is not already in the metrics list
                if (
                    column in thicket.exc_metrics
                    and column_to_append not in thicket.statsframe.exc_metrics
                ):
                    thicket.statsframe.exc_metrics.append(column_to_append)
                # check inclusive metrics
                elif (
                    column in thicket.inc_metrics
                    and column_to_append not in thicket.statsframe.inc_metrics
                ):
                    thicket.statsframe.inc_metrics.append(column_to_append)

    # columnar joined thicket object
    else:
        df_num = thicket.dataframe.select_dtypes(include=numerics)[columns]
        df = df_num.reset_index(level=1).groupby("node").quantile(percentiles)

        for idx_level, column in columns:
            calculated_percentiles = []

            # Get all the calculated values into a list for each node
            for node in pd.unique(df.reset_index()["node"]):
                calculated_percentiles.append(list(df.loc[node][(idx_level, column)]))

            # Go through each of the percentiles, and make them it's own column
            for index, percentile in enumerate(percentiles):
                column_to_append = (
                    idx_level,
                    "{}_percentiles_{}".format(column, str(int(percentile * 100))),
                )
                output_column_names.append(column_to_append)
                thicket.statsframe.dataframe[column_to_append] = [
                    x[index] for x in calculated_percentiles
                ]

                # check to see if exclusive metric
                if (
                    (idx_level, column) in thicket.exc_metrics
                    and column_to_append not in thicket.statsframe.exc_metrics
                ):
                    thicket.statsframe.exc_metrics.append(column_to_append)
                # check to see if inclusive metric
                elif (
                    (idx_level, column) in thicket.inc_metrics
                    and column_to_append not in thicket.statsframe.inc_metrics
                ):
                    thicket.statsframe.inc_metrics.append(column_to_append)

        # sort columns in index
        thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
