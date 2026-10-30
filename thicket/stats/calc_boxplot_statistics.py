# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np

from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op


@cache_stats_op
def calc_boxplot_statistics(thicket, columns=[], quartiles=[0.25, 0.5, 0.75], **kwargs):
    """Calculate boxplots lowerfence, q1, q2, q3, iqr, upperfence, and outliers for each
    node in the performance data table.

    Designed to take in a thicket, and append one or more columns to the aggregated
    statistics table for the boxplot five number summary calculations for each node.

    Arguments:
        thicket (thicket): Thicket object
        columns (list): List of columns to calculate boxplot metrics on. Note, if using
            a columnar joined thicket a list of tuples must be passed in with the format
            (column index, column name).
        quartiles (list): List containing three values between 0 and 1 to cut the
            distribution into equal probabilities.

    Returns:
        (list): returns a list of output statsframe column names
    """
    if len(quartiles) != 3:
        raise ValueError(
            "The argument quartiles takes a list containing three numbers between 0 and 1."
        )

    if len(columns) == 0:
        raise ValueError(
            "To see a list of valid columns, run 'Thicket.performance_cols'."
        )

    verify_thicket_structures(thicket.dataframe, index=["node"], columns=columns)

    output_column_names = []

    q_list = str(tuple(quartiles))

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        for col in columns:
            boxplot_dict = {
                col + "_q1" + q_list: [],
                col + "_q2" + q_list: [],
                col + "_q3" + q_list: [],
                col + "_iqr" + q_list: [],
                col + "_lowerfence" + q_list: [],
                col + "_upperfence" + q_list: [],
                col + "_outliers" + q_list: [],
            }

            # output_column_names.append(str(boxplot_dict.keys()))
            for k in boxplot_dict:
                output_column_names.append(k)

            df = thicket.dataframe.reset_index().groupby("node")
            for node, item in df:
                values = df.get_group(node)[col].tolist()

                q = np.quantile(values, quartiles)
                q1 = q[0]
                median = q[1]
                q3 = q[2]

                iqr = q3 - q1
                lower_fence = q1 - (1.5 * iqr)
                upper_fence = q3 + (1.5 * iqr)

                boxplot_dict[col + "_q1" + q_list].append(q1)
                boxplot_dict[col + "_q2" + q_list].append(median)
                boxplot_dict[col + "_q3" + q_list].append(q3)
                boxplot_dict[col + "_iqr" + q_list].append(iqr)
                boxplot_dict[col + "_lowerfence" + q_list].append(lower_fence)
                boxplot_dict[col + "_upperfence" + q_list].append(upper_fence)

                profile = []
                for i in range(0, len(values)):
                    if values[i] > upper_fence or values[i] < lower_fence:
                        profile.append(
                            thicket.dataframe.loc[node].reset_index()[
                                thicket.profile_idx_name
                            ][i]
                        )
                    else:
                        continue
                if not profile:
                    profile = []
                boxplot_dict[col + "_outliers" + q_list].append(profile)
            # check to see if exclusive metric
            if col in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.extend(list(boxplot_dict.keys()))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.extend(list(boxplot_dict.keys()))

            df_box = pd.DataFrame(
                boxplot_dict,
                index=pd.Index(
                    thicket.statsframe.dataframe.index.tolist(), name="node"
                ),
            )
            thicket.statsframe.dataframe = thicket.statsframe.dataframe.join(df_box)
    # columnar joined thicket object
    else:
        for idx, col in columns:
            boxplot_dict = {
                idx: {
                    col + "_q1" + q_list: [],
                    col + "_q2" + q_list: [],
                    col + "_q3" + q_list: [],
                    col + "_iqr" + q_list: [],
                    col + "_lowerfence" + q_list: [],
                    col + "_upperfence" + q_list: [],
                    col + "_outliers" + q_list: [],
                }
            }

            for k in boxplot_dict[idx]:
                output_column_names.append((idx, k))

            df = thicket.dataframe.reset_index().groupby("node")
            for node, item in df:
                values = df.get_group(node)[(idx, col)].tolist()

                q = np.quantile(values, quartiles)
                q1 = q[0]
                median = q[1]
                q3 = q[2]

                iqr = q3 - q1
                lower_fence = q1 - (1.5 * iqr)
                upper_fence = q3 + (1.5 * iqr)

                boxplot_dict[idx][col + "_q1" + q_list].append(q1)
                boxplot_dict[idx][col + "_q2" + q_list].append(median)
                boxplot_dict[idx][col + "_q3" + q_list].append(q3)
                boxplot_dict[idx][col + "_iqr" + q_list].append(iqr)
                boxplot_dict[idx][col + "_lowerfence" + q_list].append(lower_fence)
                boxplot_dict[idx][col + "_upperfence" + q_list].append(upper_fence)

                profile = []
                for i in range(0, len(values)):
                    if values[i] > upper_fence or values[i] < lower_fence:
                        profile.append(
                            thicket.dataframe[idx]
                            .loc[node]
                            .reset_index()[thicket.profile_idx_name][i]
                        )
                    else:
                        continue
                if not profile:
                    profile = []
                boxplot_dict[idx][col + "_outliers" + q_list].append(profile)

            create_multi_index = {
                (outerKey, innerKey): values
                for outerKey, innerDict in boxplot_dict.items()
                for innerKey, values in innerDict.items()
            }

            # check to see if exclusive metric
            if (idx, col) in thicket.exc_metrics:
                thicket.statsframe.exc_metrics.extend(list(create_multi_index.keys()))
            # check to see if inclusive metric
            else:
                thicket.statsframe.inc_metrics.extend(list(create_multi_index.keys()))

            df_box = pd.DataFrame(
                create_multi_index,
                index=pd.Index(
                    thicket.statsframe.dataframe.index.tolist(), name="node"
                ),
            )
            thicket.statsframe.dataframe = thicket.statsframe.dataframe.join(df_box)

        # sort columns in index
        # thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
