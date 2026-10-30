# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

import thicket as th
from ..utils import verify_thicket_structures
from .stats_utils import cache_stats_op
from thicket.stats import mean


def _calc_score_delta_mean_delta_stdnorm(means_1, means_2, stds_1, stds_2):
    results = []

    for i in range(len(means_1)):
        result = None
        try:
            result = (means_1[i] - means_2[i]) + (
                (stds_1[i] - stds_2[i]) / (np.abs(means_1[i] - means_2[i]))
            )
        except RuntimeWarning:
            result = np.nan
        results.append(result)

    return results


def _calc_score_delta_mean_delta_coefficient_of_variation(
    means_1,
    means_2,
    stds_1,
    stds_2,
):
    results = []

    for i in range(len(means_1)):
        result = (
            (means_1[i] - means_2[i])
            + (stds_1[i] / means_1[i])
            - (stds_2[i] / means_2[i])
        )
        results.append(result)
    return results


def _calc_bhattacharyya_score(
    thicket, columns, comparison_func, characterization_func, **kwargs
):
    stats_column_names = th.stats.bhattacharyya_distance(thicket, columns)

    # Execute characterization function
    comp_idx = characterization_func(thicket, columns=columns, **kwargs)
    comp_data = thicket.statsframe.dataframe[[comp_idx[0], comp_idx[1]]]

    # Apply comparison function to characterization output
    multiplier = comp_data.apply(
        lambda row: comparison_func(row[comp_idx[0]], row[comp_idx[1]]), axis=1
    )
    multiplier = multiplier.apply(lambda x: -1 if x else 1)

    result = thicket.statsframe.dataframe[stats_column_names[0]] * multiplier

    return result


def _calc_hellinger_score(
    thicket, columns, comparison_func, characterization_func, **kwargs
):
    stats_column_names = th.stats.hellinger_distance(thicket, columns)

    # Execute characterization function
    comp_idx = characterization_func(thicket, columns=columns, **kwargs)
    comp_data = thicket.statsframe.dataframe[[comp_idx[0], comp_idx[1]]]

    # Apply comparison function to characterization output
    multiplier = comp_data.apply(
        lambda row: comparison_func(row[comp_idx[0]], row[comp_idx[1]]), axis=1
    )
    multiplier = multiplier.apply(lambda x: -1 if x else 1)

    result = thicket.statsframe.dataframe[stats_column_names[0]] * multiplier

    return result


def _score_delta(thicket, columns, output_column_name, scoring_function):
    if isinstance(columns, list) is False:
        raise ValueError("Value passed to 'columns' must be of type list.")

    # Columns must be a tuples since we are dealing with columnar joined thickets
    for column in columns:
        if isinstance(column, tuple) is False:
            raise ValueError(
                "Columns listed in 'columns' argument must be of type tuple."
            )

    if not isinstance(thicket, th.Thicket):
        raise ValueError(
            "Value passed to 'thicket' argument must be of type thicket.Thicket."
        )

    if thicket.dataframe.columns.nlevels == 1:
        raise ValueError(
            "Value passed to 'thicket' argument must be a columnar joined thicket."
        )

    if len(columns) != 2:
        raise ValueError("Value passed to 'columns' argument must be of length 2.")

    verify_thicket_structures(thicket.dataframe, columns)

    output_column_names = []

    mean_columns = th.stats.mean(thicket, columns)
    std_columns = th.stats.std(thicket, columns)

    means_target1 = thicket.statsframe.dataframe[mean_columns[0]].to_numpy()
    means_target2 = thicket.statsframe.dataframe[mean_columns[1]].to_numpy()

    stds_target1 = thicket.statsframe.dataframe[std_columns[0]].to_numpy()
    stds_target2 = thicket.statsframe.dataframe[std_columns[1]].to_numpy()

    # Call the scoring function that the user specified
    resulting_scores = scoring_function(
        means_target1,
        means_target2,
        stds_target1,
        stds_target2,
    )
    # User can specify a column name for the statsframe, otherwise default it to:
    #   "target1_column1_target2_column2_scoreFunctionName"
    stats_frame_column_name = None

    if output_column_name is None:
        stats_frame_column_name = "{}_{}_{}_{}_{}".format(
            columns[0][0],
            columns[0][1],
            columns[1][0],
            columns[1][1],
            scoring_function.__name__,
        )
    else:
        stats_frame_column_name = output_column_name

    thicket.statsframe.dataframe["Scoring", stats_frame_column_name] = resulting_scores
    thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    output_column_names.append(("Scoring", stats_frame_column_name))

    return output_column_names


@cache_stats_op
def score_delta_mean_delta_stdnorm(thicket, columns, output_column_name=None):
    r"""
    Apply a mean difference with standard deviation difference algorithm on two
    passed columns. The passed columns must be from the performance data table.

    Designed to take in a thicket object, specified columns, an output column name, and
    append the scoring to the thicket statsframe.

    This provides a quantitative way to compare two columns in terms of both their means
    and standard deviations. Yields a single value that represents the overall difference,
    considering both their central tendency and spread.

    Arguments:
        thicket (thicket)   : Thicket object
        columns (list)      : List of hardware/timing metrics to perform scoring on. A
            columnar joined thicket is required and as such  a list of tuples must be
            passed in with the format (column index, column name).
        output_column_name (string)  : A string that assigns a name to the scoring column.

    Returns:
        (list): returns a list of output statsframe column names

    Equation:
        .. math::

            \text{result} = (\mu_1[i] - \mu_2[i]) + \frac{{\left(\sigma_1[i] - \sigma_2[i]\right)}}{{\left|\mu_1[i] - \mu_2[i]\right|}}
    \]
    """

    output_column_names = _score_delta(
        thicket,
        columns,
        output_column_name,
        _calc_score_delta_mean_delta_stdnorm,
    )

    return output_column_names


@cache_stats_op
def score_delta_mean_delta_coefficient_of_variation(
    thicket, columns, output_column_name=None
):
    r"""
    Apply a mean difference with difference spread of data algorithm on two passed columns.
    The passed columns must be from the performance data table.

    Designed to take in a thicket object, specified columns, an output column name, and
    append the scoring to the thicket statsframe.

    This provides a quantitative way to compare two columns through a measure that considers
    both the difference in means and the difference in spread of data (represented by the
    coefficient of variation).

    Arguments:
        thicket (thicket)   : Thicket object
        columns (list)      : List of hardware/timing metrics to perform scoring on. A
            columnar joined thicket is required and as such  a list of tuples must be
            passed in with the format (column index, column name).
        output_column_name  : A string that assigns a name to the scoring column.

    Returns:
        (list): returns a list of output statsframe column names

    Equation:
        .. math::

            \text{result} = (\mu_1[i] - \mu_2[i]) + \frac{{\sigma_1[i]}}{{\mu_1[i]}} - \frac{{\sigma_2[i]}}{{\mu_2[i]}}
    """

    stats_column_names = _score_delta(
        thicket,
        columns,
        output_column_name,
        _calc_score_delta_mean_delta_coefficient_of_variation,
    )

    return stats_column_names


@cache_stats_op
def score_bhattacharyya(
    thicket,
    columns,
    output_column_name=None,
    characterization_func=mean,
    comparison_func=None,
    **kwargs,
):
    r"""
    Calculate a score using the Bhattacharyya distance metric along with a characterization function.

    Designed to take in a thicket object, specified columns, an output column name, and
    append the result to the thicket statsframe.

    To form a proper scoring mechanism, the approach is divided into two steps, magnitude and signage.

    The Bhattacharyya distance metric measures the amount of overlap between two statistical samples or
    populations. It calculates the distance as a function of means and variances. It ranges from 0 to positive
    infinity, with 0 indicating complete overlap and vice versa. It only provides a scalar value which is useful
    for determining how different two samples are but provides no insight on which is preferable.

    Towards this end, a signage is required to indicate a distinction between the samples. A characterization function
    along with a comparison function is used for this.

    Arguments:
        thicket (thicket)   : Thicket object
        columns (list)      : List of hardware/timing metrics to perform computation on. A
            columnar joined thicket is required and as such  a list of tuples must be
            passed in with the format (column index, column name).
        output_column_name  : A string that assigns a name to the resulting column.
        characterization_func: A thicket stats function that calculates a preference statistic. Used in
            conjunction with a comparison function.
        comparison_func: A function used to make a binary comparison determining signage. This function must
            return a boolean value.

    Returns:
        (list): returns a list of output statsframe column names

    Equation:
        .. math::

            \text{result} = \frac{1}{4} \cdot \log \left( \frac{1}{4} \cdot \left( \frac{{\sigma_1[i]^2}}{{\sigma_2[i]^2}} + \frac{{\sigma_2[i]^2}}{{\sigma_1[i]^2}} + 2 \right) \right) + \frac{1}{4} \cdot \left( \frac{{(\mu_1[i] - \mu_2[i])^2}}{{\sigma_1[i]^2 + \sigma_2[i]^2}} \right)
    """

    if not isinstance(thicket, th.Thicket):
        raise ValueError(
            "Value passed to 'thicket' argument must be of type thicket.Thicket."
        )

    if isinstance(columns, list) is False:
        raise ValueError("Value passed to 'columns' must be of type list.")

    if len(columns) != 2:
        raise ValueError("Value passed to 'columns' must be a list of size 2.")

    verify_thicket_structures(thicket.dataframe, columns)

    output_column_names = []

    if comparison_func is None:
        comparison_func = lambda x, y: x < y  # noqa: E731

    if output_column_name is None:
        output_column_name = "bhattacharyya"

    score_column = ("Scoring", output_column_name)

    result = _calc_bhattacharyya_score(
        thicket, columns, comparison_func, characterization_func
    )

    thicket.statsframe.dataframe[score_column] = result

    output_column_names.append(score_column)

    thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names


@cache_stats_op
def score_hellinger(
    thicket,
    columns,
    output_column_name=None,
    characterization_func=mean,
    comparison_func=None,
    **kwargs,
):
    r"""
    Calculate a score using the Hellinger distance metric along with a characterization function.

    Designed to take in a thicket object, specified columns, an output column name, and
    append the result to the thicket statsframe.

    To form a proper scoring mechanism, the approach is divided into two steps, magnitude and signage.

    This provides a quantitative way to compare two columns through the Hellinger distance,
    which is used to quantify the similarity between two probability distributions. It is based
    on comparing the square roots of the probability densities rather than the probabilities
    themselves. Hellinger distance ranges from 0 to 1, with 0 indicating identical distributions
    and 1 indicating completely different distribution.

    The Hellinger distance quantifies the similarity between two probability distributions. It is based
    on comparing the square roots of the probability densities rather than the probabilities themselves.
    It ranges is based on comparing the square roots of the probability densities rather than the probabilities
    themselves. It only provides a scalar value which is useful for determining how different two samples are but provides no insight on which is preferable.

    Towards this end, a signage is required to indicate a distinction between the samples. A characterization function
    along with a comparison function is used for this.

    Arguments:
        thicket (thicket)   : Thicket object
        columns (list)      : List of hardware/timing metrics to perform computation on. A
            columnar joined thicket is required and as such  a list of tuples must be
            passed in with the format (column index, column name).
        output_column_name  : A string that assigns a name to the resulting column.
        characterization_func: A thicket stats function that calculates a preference statistic. Used in
            conjunction with a comparison function.
        comparison_func: A function with two parameters used to make a binary comparison determining signage.
            This function should return a boolean value.

    Returns:
        (list): returns a list of output statsframe column names

    Equation:
        .. math::

            \text{result} = 1 - \sqrt{\frac{{2 \sigma_1[i]\sigma_2[i]}}{{\sigma_1[i]^2 + \sigma_2[i]^2}}} \cdot \mathrm{e}^{-\frac{1}{4}\frac{{ (\mu_1[i] - \mu_2[i])^2}}{{\sigma_1[i]^2 + \sigma_2[i]^2}}}
    """
    if not isinstance(thicket, th.Thicket):
        raise ValueError(
            "Value passed to 'thicket' argument must be of type thicket.Thicket."
        )

    if isinstance(columns, list) is False:
        raise ValueError("Value passed to 'columns' must be of type list.")

    if len(columns) != 2:
        raise ValueError("Value passed to 'columns' must be a list of size 2.")

    verify_thicket_structures(thicket.dataframe, columns)

    output_column_names = []

    if comparison_func is None:
        comparison_func = lambda x, y: x < y  # noqa: E731

    if output_column_name is None:
        output_column_name = "hellinger"

    score_column = ("Scoring", output_column_name)

    result = _calc_hellinger_score(
        thicket, columns, comparison_func, characterization_func
    )

    thicket.statsframe.dataframe[score_column] = result

    output_column_names.append(score_column)

    thicket.statsframe.dataframe = thicket.statsframe.dataframe.sort_index(axis=1)

    return output_column_names
