# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import base64
from io import BytesIO
from statistics import mean

import extrap.entities as xent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extrap.entities.experiment import (
    Experiment,
)  # For some reason it errors if "Experiment" is not explicitly imported
from extrap.fileio import io_helper
from extrap.modelers.model_generator import ModelGenerator


MODEL_TAG = "_extrap-model"


class ModelWrapper:
    """Wrapper for an Extra-P model.

    Provides more convenient functions for evaluating the model at given data points,
    for writing out a string representation of the model, and for displaying (plotting)
    the model.
    """

    def __init__(self, mdl, param_name):
        self.mdl = mdl
        self.param_name = param_name  # Needed for plotting / displaying the model

    def __str__(self):
        """Display self as a function"""
        return str(self.mdl.hypothesis.function)

    def eval(self, val):
        """Evaluate function (self) at val. f(val) = result"""
        return self.mdl.hypothesis.function.evaluate(val)

    def display(self, RSS):
        """Display function

        Arguments:
            RSS (bool): whether to display Extra-P RSS on the plot
        """
        # Sort based on x values
        measures_sorted = sorted(self.mdl.measurements, key=lambda x: x.coordinate[0])

        # Scatter plot
        params = [ms.coordinate[0] for ms in measures_sorted]  # X values
        measures = [ms.value(True) for ms in measures_sorted]  # Y values

        # Line plot

        # X value plotting range. Dynamic based off what the largest/smallest values are
        x_vals = np.arange(
            params[0], 1.5 * params[-1], (params[-1] - params[0]) / 100.0
        )

        # Y values
        y_vals = [self.mdl.hypothesis.function.evaluate(x) for x in x_vals]

        plt.ioff()
        fig, ax = plt.subplots()

        # Plot line
        ax.plot(x_vals, y_vals, label=self.mdl.hypothesis.function)

        # Plot scatter
        ax.plot(params, measures, "ro", label=self.mdl.callpath)

        ax.set_xlabel(self.param_name)
        ax.set_ylabel(self.mdl.metric)
        if RSS:
            ax.text(
                x_vals[0],
                max(y_vals + measures),
                "RSS = " + self.mdl.hypothesis.RSS,
            )
        ax.legend(loc=1)

        return fig, ax


class Modeling:
    """Produce models for all the metrics across the given graphframes."""

    def __init__(self, tht, param_name, params=None, chosen_metrics=None):
        """Create a new model object.

        Adds a model column for each metric for each common frame across all the
        graphframes.

        The given list of params contains the parameters to build the models.  For
        example, MPI ranks, input sizes, and so on.

        Arguments:
            tht (Thicket): thicket object
            param_name (str): arbitrary if 'params' is being provided, otherwise name of
                the metadata column from which 'params' will be extracted
            params (list): parameters list, domain for the model
            chosen_metrics (list): metrics to be evaluated in the model, range for the
                model
        """
        self.tht = tht
        self.param_name = param_name

        # Assign param
        # Get params from metadata table
        if not params:
            self.params = self.tht.metadata[param_name].tolist()
        # params must be provided by the user
        else:
            if not isinstance(params, dict):
                raise TypeError("'params' must be provided as a dict")
            elif len(params) != len(self.tht.profile):
                raise ValueError(
                    "length of params must equal amount of profiles "
                    + len(params)
                    + "!= "
                    + len(self.tht.profile)
                )
            profile_mapping_flipped = {
                v: k for k, v in self.tht.profile_mapping.items()
            }
            for file_name, value in params.items():
                self.tht.metadata.at[
                    profile_mapping_flipped[file_name], param_name
                ] = value
            self.params = tht.metadata[param_name].tolist()
        if not chosen_metrics:
            self.chosen_metrics = self.tht.exc_metrics + self.tht.inc_metrics
        else:
            self.chosen_metrics = chosen_metrics

    def to_html(self, RSS=False):
        def model_to_img_html(model_obj):
            fig, ax = model_obj.display(RSS)
            figfile = BytesIO()
            fig.savefig(figfile, format="jpg", transparent=False)
            figfile.seek(0)
            figdata_jpg = base64.b64encode(figfile.getvalue()).decode()
            imgstr = '<img src="data:image/jpg;base64,{}" />'.format(figdata_jpg)
            plt.close(fig)
            return imgstr

        frm_dict = {met + MODEL_TAG: model_to_img_html for met in self.chosen_metrics}

        # Subset of the aggregated statistics table with only the Extra-P columns selected
        return self.tht.statsframe.dataframe[
            [met + MODEL_TAG for met in self.chosen_metrics]
        ].to_html(escape=False, formatters=frm_dict)

    def _add_extrap_statistics(self, node, metric):
        """Insert the Extra-P hypothesis function statistics into the aggregated
            statistics table. Has to be called after "produce_models".

        Arguments:
            node (hatchet.Node): The node for which statistics should be calculated
            metric (str): The metric for which statistics should be calculated
        """
        hypothesis_fn = self.tht.statsframe.dataframe.at[
            node, metric + MODEL_TAG
        ].mdl.hypothesis

        self.tht.statsframe.dataframe.at[
            node, metric + "_RSS" + MODEL_TAG
        ] = hypothesis_fn.RSS
        self.tht.statsframe.dataframe.at[
            node, metric + "_rRSS" + MODEL_TAG
        ] = hypothesis_fn.rRSS
        self.tht.statsframe.dataframe.at[
            node, metric + "_SMAPE" + MODEL_TAG
        ] = hypothesis_fn.SMAPE
        self.tht.statsframe.dataframe.at[
            node, metric + "_AR2" + MODEL_TAG
        ] = hypothesis_fn.AR2
        self.tht.statsframe.dataframe.at[
            node, metric + "_RE" + MODEL_TAG
        ] = hypothesis_fn.RE

    def produce_models(self, agg_func=mean, add_stats=True):
        """Produces an Extra-P model. Models are generated by calling Extra-P's
            ModelGenerator.

        Arguments:
            agg_func (function): aggregation function to apply to multi-dimensional
                measurement values. Extra-P v4.0.4 applies mean by default so that is
                set here for clarity.
            add_stats (bool): Option to add hypothesis function statistics to the
                aggregated statistics table
        """
        # Setup domain values one time. Have to match ordering with range
        # values (i.e. performance data table profile ordering)
        param_coords = []  # default coordinates for all profiles

        # Mapping from metadata profiles to the parameter
        meta_param_mapping = self.tht.metadata[self.param_name].to_dict()

        # Flipped version of mapping dictionary
        meta_param_mapping_flipped = dict(
            [(value, key) for key, value in meta_param_mapping.items()]
        )

        # Ordering of profiles in the performance data table
        ensemble_profile_ordering = list(self.tht.dataframe.index.unique(level=1))

        # Append coordinates in order
        for profile in ensemble_profile_ordering:
            param_coords.append(
                xent.coordinate.Coordinate(float(meta_param_mapping[profile]))
            )

        # Iterate over nodes (outer index)
        for node, single_node_df in self.tht.dataframe.groupby(level=0):
            # Start experiment
            exp = Experiment()
            exp.add_parameter(xent.parameter.Parameter(self.param_name))

            # For all chosen metrics
            for met in self.chosen_metrics:
                measures = []
                # Iterate over profiles (secondary index)
                for profile, single_prof_df in single_node_df.groupby(level=1):
                    measures.append(single_prof_df[met].tolist())
                # Apply aggregation function to measurements
                for i in range(len(measures)):
                    # Only apply aggregation function if more that one value
                    if isinstance(measures[i], list):
                        measures[i] = agg_func(measures[i])
                # Add coordinates (points at which measurements were taken) for the
                # default case
                if len(measures) == len(param_coords):
                    exp.coordinates.extend(param_coords)
                # Handle case where profile(s) do not contain a measurement for the
                # current node
                elif len(measures) < len(param_coords):
                    param_coords_subset = []
                    df_index = self.tht.dataframe.loc[node, met].index
                    for coord in param_coords:
                        profile = meta_param_mapping_flipped[coord._values[0]]
                        if df_index.isin([profile], level=0).any():
                            param_coords_subset.append(coord)
                        else:
                            print(
                                "(Coordinate removed) Measurement at ("
                                + profile
                                + "): "
                                + meta_param_mapping[profile]
                                + "DNE for node "
                                + node
                            )
                    exp.coordinates.extend(param_coords_subset)

                # Create metric object
                metric_obj = xent.metric.Metric(met)
                exp.add_metric(xent.metric.Metric(met))
                # Create callpath object and call tree
                cpath = xent.callpath.Callpath(node.frame["name"])
                exp.add_callpath(cpath)
                exp.call_tree = io_helper.create_call_tree(exp.callpaths)
                # Add measurement objects to experiment
                for coord, measurement in zip(exp.coordinates, measures):
                    measurement_obj = xent.measurement.Measurement(
                        coord, cpath, metric_obj, measurement
                    )
                    exp.add_measurement(measurement_obj)
                # Sanity check
                io_helper.validate_experiment(exp)
                # Generate model
                model_gen = ModelGenerator(exp)
                model_gen.model_all()
                mkey = (cpath, metric_obj)
                self.tht.statsframe.dataframe.at[node, met + MODEL_TAG] = ModelWrapper(
                    model_gen.models[mkey], self.param_name
                )
                # Add statistics to aggregated statistics table
                if add_stats:
                    self._add_extrap_statistics(node, met)

    def _componentize_function(model_object):
        """Componentize one Extra-P modeling object into a dictionary of its parts

        Arguments:
            model_object (ModelWrapper): Thicket ModelWrapper Extra-P modeling object

        Returns:
            (dict): dictionary of the ModelWrapper's hypothesis function parts
        """
        # Dictionary of variables mapped to coefficients
        term_dict = {}
        # Model object hypothesis function
        fnc = model_object.mdl.hypothesis.function
        # Constant "c" column
        term_dict["c"] = fnc.constant_coefficient

        # Terms of form "coefficient * variables"
        for term in fnc.compound_terms:
            # Join variables of the same term together
            variable_column = " * ".join(t.to_string() for t in term.simple_terms)

            term_dict[variable_column] = term.coefficient

        return term_dict

    def componentize_statsframe(self, columns=None):
        """Componentize multiple Extra-P modeling objects in the aggregated statistics
        table

        Arguments:
            column (list): list of column names in the aggregated statistics table to
                componentize. Values must be of type 'thicket.model_extrap.ModelWrapper'.
        """
        # Use all Extra-P columns
        if columns is None:
            columns = [
                col
                for col in self.tht.statsframe.dataframe
                if isinstance(self.tht.statsframe.dataframe[col].iloc[0], ModelWrapper)
            ]

        # Error checking
        for c in columns:
            if c not in self.tht.statsframe.dataframe.columns:
                raise ValueError(
                    "column " + c + " is not in the aggregated statistics table."
                )
            elif not isinstance(self.tht.statsframe.dataframe[c].iloc[0], ModelWrapper):
                raise TypeError(
                    "column "
                    + c
                    + " is not the right type (thicket.model_extrap.ModelWrapper)."
                )

        # Process each column
        all_dfs = []
        for col in columns:
            # Get list of components for this column
            components = [
                Modeling._componentize_function(model_obj)
                for model_obj in self.tht.statsframe.dataframe[col]
            ]

            # Component dataframe
            comp_df = pd.DataFrame(
                data=components, index=self.tht.statsframe.dataframe.index
            )

            # Add column name as index level
            comp_df.columns = pd.MultiIndex.from_product(
                [[col], comp_df.columns.to_list()]
            )
            all_dfs.append(comp_df)

        # Concatenate dataframes horizontally
        all_dfs.insert(0, self.tht.statsframe.dataframe)
        self.tht.statsframe.dataframe = pd.concat(all_dfs, axis=1)
