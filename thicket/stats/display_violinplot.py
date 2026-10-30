# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import hatchet as ht

import thicket as th
from .percentiles import percentiles
from ..utils import verify_thicket_structures


def _column_name_mapper(current_cols):
    """
    Internal function that returns a string representation of 'current_cols'.

    Arguments:
        current_cols (list)   : List of of current column names.

    Returns:
        (string object)   : String representing:
            (current_cols) : 'current_cols' in string format
            (current_cols[0]): 'current_cols[0]' in string format
    """
    if current_cols[0] in ["node", "name"]:
        return current_cols[0]

    return str(current_cols)


def _add_percentile_lines(
    graph,
    graphType,
    thickets,
    nodes,
    columns,
    x_order,
    percentiles_vals,
    line_styles=None,
    line_colors=None,
):
    if isinstance(percentiles_vals, list):
        if isinstance(line_styles, list) is True:
            if len(percentiles_vals) != len(line_styles):
                raise ValueError(
                    "Length of line styles does not match length of percentiles"
                )
        elif line_styles is None:
            line_styles = ["--"] * len(percentiles_vals)
        elif isinstance(line_styles, str):
            line_styles = [line_styles] * len(percentiles_vals)
        else:
            raise ValueError("line_styles must be either None, list, or a str")

        if isinstance(line_colors, list) is True:
            if len(percentiles_vals) != len(line_colors):
                raise ValueError(
                    "Length of line colors does not match length of percentiles"
                )
        # Default line color
        elif line_colors is None:
            line_colors = ["black"] * len(percentiles_vals)
        elif isinstance(line_colors, str):
            line_colors = [line_colors] * len(percentiles_vals)
        else:
            raise ValueError("line_colors must be either None, list, or a str")
    # A single value was passed into percentiles, line color/style must then be either a single
    #   str value, or None. Otherwise throw an error
    elif isinstance(percentiles_vals, float):
        if isinstance(line_styles, str) is True:
            line_styles = [line_styles]
        elif line_styles is None:
            line_styles = ["--"]
        else:
            raise ValueError(
                "Percentiles was specified as a single value, line_style must be either a str, or None"
            )

        if isinstance(line_colors, str) is True:
            line_colors = [line_colors]
        elif line_colors is None:
            line_colors = ["black"]
        else:
            raise ValueError(
                "Percentiles was specified as a single value, line_colors must be either a str, or None"
            )
        percentiles_vals = [percentiles_vals]
    else:
        raise ValueError(
            "percentiles_vals must be either a list of floats, or a single float value!"
        )

    violin_idx = -1

    # Check the type of the graph being developed.
    # If this function was called from display_violinplot(...)
    # Graph type should be "NODE", and we set dummy dictionaries
    # of the inputs in order to work with the display_violinplot_thickets(...)
    # structure of this function
    if graphType == "NODE":
        thickets = {"IGNORE": thickets}
        nodes = {"IGNORE": nodes}
        columns = {"IGNORE": columns}
        x_order = ["IGNORE"]

    for thicket_key in x_order:
        thicket = thickets[thicket_key]
        nodes_in = nodes[thicket_key]
        columns_in = columns[thicket_key]

        # Make a list of a singular node since display_violinplot_thicket(...) takes
        # in a dictionary where the value is a singluar Node. display_violinplot(...)
        # Takes in a list of thickets, so the conversion doesn't need to happen.
        if graphType == "THICKET":
            nodes_in = [nodes_in]

        for node in nodes_in:
            for column in columns_in:
                violin_idx += 1
                for idx, percentile in enumerate(percentiles_vals):
                    stats_column = None
                    # Columnar joined thickets
                    if isinstance(column, tuple):
                        stats_column = (
                            str(column[0]),
                            "{}_percentiles_{}".format(
                                column[1], int(percentile * 100)
                            ),
                        )
                    # Non-columnar joined
                    else:
                        stats_column = "{}_percentiles_{}".format(
                            column, int(percentile * 100)
                        )
                    # Call percentile(...) if the percentile value for the column has not been calculated already
                    if (
                        stats_column
                        not in thicket.statsframe.dataframe.columns.tolist()
                    ):
                        percentiles(thicket, [column], [percentile])

                    percentile_value = thicket.statsframe.dataframe[stats_column][node]

                    # Customizing plotting code was taken from a stack overflow post:
                    #    https://stackoverflow.com/a/67333021

                    # Plot line
                    added_line = graph.axhline(
                        y=percentile_value,
                        color=line_colors[idx],
                        linestyle=line_styles[idx],
                    )
                    # Get the violin structure
                    patch = mpl.patches.PathPatch(
                        graph.collections[violin_idx].get_paths()[0],
                        transform=graph.transData,
                    )
                    # Clip line to the violin structure
                    added_line.set_clip_path(patch)

    return graph


def display_violinplot(
    thicket,
    nodes=None,
    columns=None,
    percentiles=None,
    percentile_linestyles=None,
    percentile_colors=None,
    **kwargs,
):
    """
    Display a violinplot for each user passed node(s) and column(s). The passed nodes
    and columns must be from the performance data table.

    Designed to take in a thicket, and display a plot with one or more violinplots
    depending on the number of nodes and columns passed.

    Arguments:
        thicket (thicket)   : Thicket object
        nodes   (list)      : List of nodes to view on the x-axis
        columns (list)      : List of hardware/timing metrics to view on the y-axis. Note, if
            using a columnar joined thicket a list of tuples must be passed in with the
            format (column index, column name).
        percentiles (list)  : List of percentile values to plot for each violin plot.
            Each value must be between [0.0, 1.0]. If the percentile value is not on the statistic table,
            the percentile value will be calculated, and put on the statsframe
        percentile_linestyles (list): List of line styles for the percentiles. If not provided, the default
            line styles will be "--". If the list does not specify styles for each of the percentiles, the missing
            styles will default to "--"
        percentile_colors (list): List of colors to apply for percentile lines. The default value is "black",
            If the list does not specify colors for each percentile, the missing colors will default to "black"
        **kwargs: Arguments that can be passed to seaborns plotting function

    Returns:
        (matplotlib Axes): Object for managing violinplot.
    """
    if columns is None or nodes is None:
        raise ValueError(
            "Both 'nodes' and 'columns' must be provided. To see a list of valid columns, run 'Thicket.performance_cols'.",
        )

    if not isinstance(thicket, th.Thicket):
        raise ValueError(
            "Value passed to 'thicket' argument must be of type thicket.Thicket."
        )

    if not isinstance(nodes, list):
        raise ValueError("Value passed to 'nodes' argument must be of type list.")

    if not isinstance(columns, list):
        raise ValueError("Value passed to 'columns' argument must be of type list.")

    for node in nodes:
        if not isinstance(node, ht.node.Node):
            raise ValueError(
                "Value passed to 'node' argument must be of type hatchet.node.Node."
            )

    verify_thicket_structures(thicket.dataframe, index=["node"], columns=columns)

    filtered_df = None

    # thicket object without columnar index
    if thicket.dataframe.columns.nlevels == 1:
        df = pd.melt(
            thicket.dataframe.reset_index(),
            id_vars=["node", "name"],
            value_vars=columns,
            var_name="Performance counter",
            value_name=" ",
        )

        position = []
        for node in nodes:
            idx = df.index[df["node"] == node]
            for pos in idx:
                position.append(pos)

        # rename columns such that the x-axis label is "node" and not "name", tick marks
        # will be node names
        filtered_df = df.loc[position].rename(
            columns={"node": "hatchet node", "name": "node"}
        )
    # columnar joined thicket object
    else:
        cols = [str(c) for c in columns]
        df_subset = thicket.dataframe[[("name", ""), *columns]].reset_index()
        df_subset.columns = df_subset.columns.to_flat_index().map(_column_name_mapper)
        df_subset["name"] = thicket.dataframe["name"].tolist()
        df = pd.melt(
            df_subset,
            id_vars=["node", "name"],
            value_vars=cols,
            var_name="Performance counter",
            value_name=" ",
        )

        position = []
        for node in nodes:
            idx = df.index[df["node"] == node]
            for pos in idx:
                position.append(pos)

        # rename columns such that the x-axis label is "node" and not "name", tick marks
        # will be node names
        filtered_df = df.loc[position].rename(
            columns={"node": "hatchet node", "name": "node"}
        )

    graph = None

    if len(columns) > 1:
        graph = sns.violinplot(
            data=filtered_df, x="node", y=" ", hue="Performance counter", **kwargs
        )
    else:
        graph = sns.violinplot(data=filtered_df, x="node", y=" ", **kwargs)
    # User specified percentile value lines to plot
    if percentiles is not None:
        return _add_percentile_lines(
            graph,
            "NODE",
            thicket,
            nodes,
            columns,
            [],
            percentiles,
            percentile_linestyles,
            percentile_colors,
        )
    # User did not specify percentiles, just return violinplot
    else:
        if len(columns) > 1:
            return sns.violinplot(
                data=filtered_df, x="node", y=" ", hue="Performance counter", **kwargs
            )
        else:
            return sns.violinplot(data=filtered_df, x="node", y=" ", **kwargs)


def display_violinplot_thicket(
    thickets,
    nodes=None,
    columns=None,
    x_order=None,
    percentiles=[],
    percentile_linestyles=[],
    percentile_colors=[],
    **kwargs,
):
    """
    Display violinplots for each thicket passed in, where each thicket will have
    a corresponding node specified in nodes. For each node, a violin will be plotted for
    each column specified in the columns dictionary.

    Arguments:
        thickets (dict)  : Dictionary of thickets. The thickets in the dictionary must either be
            all columnar joined thickets, or all non-columnar joined thickets, but not a mix of both.
        nodes    (dict)  : Dictionary of a node to plot for each thicket in the thickets
            dictionary. The keys must match for both the thickets dictionary and the nodes dictionary
        columns  (dict)  : Dictionary of columns to plot for each thicket in the thickets
            dictionary. The keys must match for both the thickets dictionary and the columns dictionary
        x_order (list) : List of keys to use for plotting order. If this is none, the thickets that get
            plotted will be in whatever order thickets.keys() returns.
        percentiles (list)  : List of percentile values to plot for each violin plot.
            Each value must be between [0.0, 1.0]. If the percentile value is not on the statistic table,
            the percentile value will be calculated, and put on the statsframe. An internal call to percentiles(...)
            will be done for you.
        percentile_linestyles (list): List of line styles for the percentiles. If not provided, the default
            line styles will be "--". If the list does not specify styles for each of the percentiles, the missing
            styles will default to "--"
        percentile_colors (list): List of colors to apply for percentile line. The default value is "black",
            If the list does not specify colors for each percentile, the missing colors will default to "black"
        **kwargs: Arguments that can be passed to seaborns violinplot function

    Returns:
        (matplotlib Axes): Object for managing violinplot.
    """
    # Ensure thickets, nodes, and columns are all dictionaries
    if not isinstance(thickets, dict):
        raise ValueError("'thickets' argument must be a dictionary of thickets.")

    if not isinstance(nodes, dict):
        raise ValueError(
            "'nodes' argument must be a dictionary, specifying a node for each Thicket passed in."
        )

    if isinstance(columns, dict) is False:
        raise ValueError(
            "'columns' argument must be a dictionary, specifying columns for each Thicket passed in."
        )

    if all(isinstance(c, list) for c in columns.values()) is False:
        raise ValueError(
            "'columns' argument dictionary must have list of columns as values."
        )

    # Ensure that if x_order is not None, that it is a list!
    if x_order is not None and isinstance(x_order, list) is False:
        raise ValueError("'x_order' argument must be a list of keys.")

    # Ensure equal number of thickets and corresponding nodes and columns
    if len(thickets) != len(nodes):
        raise ValueError(
            "Length of 'nodes' argument must match length of 'thickets' argument."
        )

    if len(thickets) != len(columns):
        raise ValueError(
            "Length of 'columns' argument must match length of 'thickets' argument."
        )

    # Verify that each node in the nodes lists are node types
    if all(isinstance(n, ht.node.Node) for n in list(nodes.values())) is False:
        raise ValueError(
            "Value(s) passed to 'nodes' argument must be of type hatchet.node.Node."
        )

    # Verify that all nodes passed are of the same type and name
    if len(set([n.frame for n in nodes.values()])) != 1:
        raise ValueError(
            "Value(s) passed to 'nodes' argument must be of same type and name."
        )

    # the keys must match the thickets dictionary keys
    if set(thickets.keys()) != set(nodes.keys()):
        raise ValueError(
            "Keys in 'nodes' argument dictionary do not match keys in 'thickets' argument dictionary."
        )
    if set(thickets.keys()) != set(columns.keys()):
        raise ValueError(
            "Keys in 'columns' argument dictionary do not match keys in 'thickets' argument dictionary."
        )
    if x_order is not None and set(thickets.keys()) != set(x_order):
        raise ValueError(
            "Keys listed in 'x_order' argument do not match keys in 'thickets' argument dictionary."
        )

    # Check to see if ALL the keys across dictionaries are the same, and if x_order is not None

    # Now check that all the thickets are either all columnar joined thickets, OR, they are all non-columnar joined thickets
    # We should not be comparing thickets that are not of the same type.
    columnar_joined_found = False
    noncolumnar_joined_found = False

    for thicket in thickets.values():
        if thicket.dataframe.columns.nlevels == 1:
            noncolumnar_joined_found = True
        elif thicket.dataframe.columns.nlevels == 2:
            columnar_joined_found = True

        if noncolumnar_joined_found and columnar_joined_found:
            raise ValueError(
                "'thickets' argument dictionary can not contain both columnar joined thickets and non-columnar joined thickets."
            )

    # Verify each thicket has corresponding columns
    for idx, key in enumerate(thickets.keys()):
        verify_thicket_structures(
            thickets[key].dataframe, index=["node"], columns=columns[key]
        )

    # Get thicket key order
    x_order = list(thickets.keys()) if x_order is None else x_order

    # Holds the final dataframe for each thicket that we will plot
    filtered_dfs = [None] * len(thickets)

    # Holds the intermediate dataframe for each thicket
    sub_dataframe = None

    for thicket_idx, key in enumerate(x_order):
        thicket = thickets[key]
        node = nodes[key]
        in_columns = columns[key]

        # non-columnar joined thicket
        if thicket.dataframe.columns.nlevels == 1:
            melted_df = pd.melt(
                thicket.dataframe.reset_index(),
                id_vars=["node", "name"],
                value_vars=in_columns,
                var_name="Performance counter",
                value_name=" ",
            )

            idx_c = melted_df.index[melted_df["node"] == node]
            position = [p for p in idx_c]
            filtered_dfs[thicket_idx] = melted_df.loc[position].rename(
                columns={"node": "hatchet node", "name": "node"}
            )
            thicket_name = [str(key)] * len(filtered_dfs[thicket_idx]["node"])
            filtered_dfs[thicket_idx]["node"] = thicket_name
        # Columnar joined thicket
        else:
            cols = [str(c) for c in in_columns]
            sub_dataframe = thicket.dataframe[[("name", ""), *in_columns]].reset_index()
            sub_dataframe.columns = sub_dataframe.columns.to_flat_index().map(
                _column_name_mapper
            )
            sub_dataframe["name"] = thicket.dataframe["name"].tolist()

            melted_df = pd.melt(
                sub_dataframe,
                id_vars=["node", "name"],
                value_vars=cols,
                var_name="Performance counter",
                value_name=" ",
            )

            idx_c = melted_df.index[melted_df["node"] == node]
            position = [p for p in idx_c]
            filtered_dfs[thicket_idx] = melted_df.loc[position].rename(
                columns={"node": "hatchet node", "name": "node"}
            )
            thicket_name = [str(key)] * len(filtered_dfs[thicket_idx]["node"])
            filtered_dfs[thicket_idx]["node"] = thicket_name

    master_df = pd.concat(filtered_dfs, ignore_index=True)
    master_df = master_df.rename(columns={"node": "thicket"})

    graph = sns.violinplot(
        data=master_df,
        x="thicket",
        y=" ",
        hue="Performance counter",
        inner=None,
        **kwargs,
    )

    # No percentiles to plot!
    if len(percentiles) == 0:
        return graph
    # Percentiles need to be plotted
    else:
        return _add_percentile_lines(
            graph,
            "THICKET",
            thickets,
            nodes,
            columns,
            x_order,
            percentiles,
            percentile_linestyles,
            percentile_colors,
        )
