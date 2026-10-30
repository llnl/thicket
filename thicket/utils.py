# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from collections import Counter, OrderedDict, defaultdict
import warnings

import numpy as np
import pandas as pd

from thicket import helpers


# Define custom errors for external checks
class DuplicateIndexError(IndexError):
    pass


class DuplicateValueError(ValueError):
    pass


class MissingHNIDError(ValueError):
    pass


class InvalidNameError(ValueError):
    pass


def check_same_frame(n1, n2):
    if n1.frame != n2.frame:
        raise ValueError(
            f"Cannot replace the node. Frames do not match: {n1.frame} != {n2.frame}"
        )


def check_duplicate_metadata_key(thickets, metadata_key):
    """Check for duplicate values in the metadata of a list of Thickets for column 'metadata_key'.

    Arguments:
        thickets (list): list of Thickets to check
        metadata_key (str): metadata key to check for duplicates

    Returns:
        Raises an error if any duplicate metadata values are found
    """
    duplicates_dict = defaultdict(list)
    for i in range(len(thickets)):
        for j in range(i, len(thickets)):
            if i != j:
                m1 = set(thickets[i].metadata[metadata_key])
                m2 = set(thickets[j].metadata[metadata_key])
                duplicates = m1.intersection(m2)
                if len(duplicates) > 0:
                    duplicates_dict[(i, j)].append(duplicates)

    if len(duplicates_dict) > 0:
        err_str = ""
        for (i, j), v in duplicates_dict.items():
            err_str += f"thickets[{i}].metadata['{metadata_key}'] and thickets[{j}].metadata['{metadata_key}'] have the same values: {v}\n"
        raise DuplicateValueError(
            f"Different Thicket.metadata[metadata_key] may not contain duplicate values.\n{err_str}"
        )


def validate_dataframe(df):
    """Check validity of a Thicket DataFrame."""

    def _check_duplicate_inner_idx(df):
        """Check for duplicate values in the innermost indices."""
        for node in set(df.index.get_level_values("node")):
            inner_idx_values = sorted(df.loc[node].index.tolist())
            counts = Counter(inner_idx_values)
            duplicates = [item for item, count in counts.items() if count > 1]
            if len(duplicates) > 0:
                raise DuplicateIndexError(
                    f"Duplicate indices found in DataFrame index.\n\t{duplicates}"
                )

    def _check_missing_hnid(df):
        """Check if there are missing hatchet nid's."""
        i = 0
        set_of_nodes = set(df.index.get_level_values("node"))
        for node in set_of_nodes:
            if hash(node) != i:
                raise MissingHNIDError(
                    f"The Thicket.dataframe's index is either not sorted or has a missing node. {hash(node)} ({node}) != {i}"
                )
            i += 1

    def _validate_name_column(df):
        """Check if all of the values in a node's name column are either its name or missing."""
        for node in set(df.index.get_level_values("node")):
            names = set(df.loc[node]["name"].tolist())
            node_name = node.frame["name"]
            for name in names:
                if name != node_name and not pd.isna(name):
                    raise InvalidNameError(
                        f"Value in the Thicket.dataframe's 'name' column is not valid. {name} != {node_name}"
                    )

    _check_duplicate_inner_idx(df)
    _check_missing_hnid(df)
    _validate_name_column(df)


def validate_profile(tk):
    """Check validity of Thicket objects that rely on profiles. Thicket.dataframe, Thicket.metadata, Thicket.profile, Thicket.profile_mapping."""

    def _validate_multiindex_column(tk):
        """Check if the Thicket.dataframe and Thicket.metadata have the same column structure if MultiIndex present in either. Returns True if valid, raises ValueError if not valid."""
        if isinstance(tk.dataframe.columns, pd.MultiIndex) or isinstance(
            tk.metadata.columns, pd.MultiIndex
        ):
            if not isinstance(tk.dataframe.columns, pd.MultiIndex) or not isinstance(
                tk.metadata.columns, pd.MultiIndex
            ):
                raise ValueError(
                    "The Thicket.dataframe and Thicket.metadata must both have a MultiIndex column structure if MultiIndex present in either."
                )
        return True

    def _validate_all_same(tk):
        df_profs = set(tk.dataframe.index.droplevel(level="node"))
        meta_profs = set(tk.metadata.index)
        profs = set(tk.profile)
        pm_profs = set(tk.profile_mapping.keys())

        if isinstance(tk.dataframe.columns, pd.MultiIndex) and isinstance(
            tk.metadata.columns, pd.MultiIndex
        ):
            # Set of profiles that exist in the dataframe index
            df_profs = set(
                p
                for p in profs
                if any(dfp in helpers._powerset_from_tuple(p) for dfp in df_profs)
            )
            # Set of profiles that exist in the metadata index
            meta_profs = set(
                p
                for p in profs
                if any(mp in helpers._powerset_from_tuple(p) for mp in meta_profs)
            )

        if not df_profs == meta_profs == profs == pm_profs:
            raise ValueError(
                f"Profiles in the Thicket.dataframe, Thicket.metadata, Thicket.profile, and Thicket.profile_mapping are not the same. {df_profs} != {meta_profs} != {profs} != {pm_profs}."
            )

    def _validate_no_duplicates(tk):
        if len(tk.profile) != len(set(tk.profile)):
            raise ValueError("Profiles in the Thicket.profile must be unique.")
        elif len(tk.profile_mapping) != len(set(tk.profile_mapping)):
            raise ValueError("Profiles in the Thicket.profile_mapping must be unique.")
        elif len(tk.metadata.index) != len(set(tk.metadata.index)):
            raise ValueError("Profiles in the Thicket.metadata must be unique.")

        return True

    _validate_multiindex_column(tk)
    _validate_all_same(tk)
    _validate_no_duplicates(tk)


def verify_sorted_profile(thicket_component):
    """Assertion to check if profiles are sorted in a thicket dataframe

    Arguments:
        thicket_component (DataFrame): component of thicket to check
    """
    profile_index_values = list(
        OrderedDict.fromkeys(
            thicket_component.index.get_level_values(
                thicket_component.index.nlevels - 1
            )  # Innermost index
        )
    )
    if profile_index_values != sorted(profile_index_values):
        raise ValueError(
            "The profiles in this dataframe must be sorted. Try 'pandas.DataFrame.sort_index'."
        )


def verify_thicket_structures(thicket_component, columns=[], index=[]):
    """Assertion for missing input requirements to execute thicket functions.

    Arguments:
        thicket_components (DataFrame): component of thicket to check
        columns (list): list of columns to check for in the component
        index (list): list of index levels to check for in the component

    Returns:
        Raises an error if any columns or index levels are not in component, continues
            program if all columns and index levels are in component
    """
    if not isinstance(columns, list):
        raise RuntimeError("columns= must be specified as a list")
    if not isinstance(index, list):
        raise RuntimeError("index= must be specified as a list")

    # collect component columns and index
    component_columns = thicket_component.columns.tolist()
    component_index = thicket_component.index.names

    # check existence of columns and index
    column_result = all(elem in component_columns for elem in columns)
    index_result = all(elem in component_index for elem in index)

    # store missing columns or index if they exist
    missing_columns = list(set(columns).difference(component_columns))
    missing_index = list(set(index).difference(component_index))

    if not column_result and not index_result:
        raise RuntimeError(
            "Specified column(s) not found: "
            + str(missing_columns)
            + "\nMissing index level(s): "
            + str(missing_index)
            + " required for the function"
        )
    elif not column_result and index_result:
        raise RuntimeError("Specified column(s) not found: " + str(missing_columns))
    elif column_result and not index_result:
        raise RuntimeError(
            "Missing index level(s): "
            + str(missing_index)
            + " required for the function"
        )


def validate_nodes(tk):
    """Check if node objects match between Thicket.graph, Thicket.dataframe, and Thicket.statsframe.dataframe."""

    components_dict = {
        "Thicket.graph": {id(n): n for n in tk.graph.traverse()},
        "Thicket.dataframe": {
            id(n): n for n in tk.dataframe.index.get_level_values("node")
        },
        "Thicket.statsframe.dataframe": {
            id(n): n for n in tk.statsframe.dataframe.index
        },
    }

    set_list = [set(component.keys()) for component in components_dict.values()]
    difference = set.union(*set_list) - set.intersection(*set_list)

    debug_str = ""
    for name, node_dict in components_dict.items():
        debug_str += name + ":\n"
        for node_id in node_dict:
            if node_id in difference:
                debug_str += (
                    f"\t{hash(node_dict[node_id])} {node_dict[node_id]} {node_id}\n"
                )

    if len(difference) > 0:
        raise ValueError(
            f"Node objects do not match between Thicket.graph, Thicket.dataframe, and Thicket.statsframe.dataframe.\n{debug_str}"
        )


def _fill_perfdata(df):
    """Create full index for DataFrame and fill created rows with missing values (NaN).

    Arguments:
        df (DataFrame): DataFrame to fill missing rows in

    Returns:
        (DataFrame): filled DataFrame
    """
    new_df = df.copy(deep=True)
    try:
        # Value used to fill new rows
        fill_value = np.nan
        # Fill missing rows in dataframe with NaN's. String columns keep their
        # dtype, which also uses NaN as its missing value.
        new_df = new_df.reindex(
            pd.MultiIndex.from_product(new_df.index.levels),
            fill_value=fill_value,
        )
    except ValueError as e:
        estr = str(e)
        if estr == "cannot handle a non-unique multi-index!":
            warnings.warn(
                "Non-unique multi-index for DataFrame in _fill_perfdata. Cannot Fill missing rows.",
                RuntimeWarning,
            )
        else:
            raise

    return new_df
