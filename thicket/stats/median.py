# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op


@cache_stats_op
def median(thicket, columns=None):
    """Calculate the median for each node in the performance data table.

    Designed to take in a thicket, and append one or more columns to the
    aggregated statistics table for the median calculation for each node.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to perform median calculation
            on. Note, if using a columnar joined thicket a list of tuples must be passed
            in with the format (column index, column name).

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
        df = thicket.dataframe[columns].reset_index().groupby("node").agg("median")
        for column in columns:
            output_column_names.append(column + "_median")
            thicket.statsframe.dataframe[column + "_median"] = df[column]
            # check to see if exclusive metric
            if column in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append(column + "_median")
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append(column + "_median")

    # columnar joined thicket object
    else:
        df = (
            thicket.dataframe[columns]
            .reset_index(level=1)
            .groupby("node")
            .agg("median")
        )
        for idx, column in columns:
            output_column_names.append(str((idx, column + "_median")))
            thicket.statsframe.dataframe[(idx, column + "_median")] = df[(idx, column)]
            # check to see if exclusive metric
            if (idx, column) in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append((idx, column + "_median"))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append((idx, column + "_median"))

        # sort columns in index
        thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
