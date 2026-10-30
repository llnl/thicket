# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op


@cache_stats_op
def minimum(thicket, columns=None):
    """Determine the minimum for each node in the performance data table.

    Designed to take in a thicket, and append one or more columns to the aggregated
    statistics table for the minimum value for each node.

    The minimum is the lowest observation for a node and its associated profiles.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of hardware/timing metrics to determine minimum value for.
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
        df = thicket.dataframe[columns].reset_index().groupby("node").agg("min")
        for column in columns:
            output_column_names.append(column + "_min")
            thicket.statsframe.dataframe[column + "_min"] = df[column]
            # check to see if exclusive metric
            if column in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append(column + "_min")
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append(column + "_min")
    # columnar joined thicket object
    else:
        df = thicket.dataframe[columns].reset_index(level=1).groupby("node").agg("min")
        for idx, column in columns:
            output_column_names.append((idx, column + "_min"))
            thicket.statsframe.dataframe[(idx, column + "_min")] = df[(idx, column)]
            # check to see if exclusive metric
            if (idx, column) in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.append((idx, column + "_min"))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.append((idx, column + "_min"))

        # sort columns in index
        thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
