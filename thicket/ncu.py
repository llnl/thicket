# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from difflib import SequenceMatcher
import re

from hatchet import QueryMatcher
import pandas as pd
from tqdm import tqdm


def _match_call_trace_regex(
    kernel_call_trace, demangled_kernel_name, debug, action=None
):
    """Use the NCU call trace to regex match the kernel name from the demangled
    kernel string. Returns the matched kernel string, if match is possible.

    Arguments:
        kernel_call_trace (list): List of strings from NCU representing the call trace
        demangled_kernel_name (str): Demangled kernel name from NCU
        debug (bool): Print debug statements
        action (ncu_report.IAction): NCU action object
    """
    # Call trace with last element removed (last elem usually not useful for matching)
    temp_call_trace = kernel_call_trace[:-1]
    # Special case to match "cub" kernels
    if "cub" in demangled_kernel_name:
        call_trace_str = "cub"
        # Replace substrings that may cause mismatch
        demangled_kernel_name = demangled_kernel_name.replace("(bool)1", "true")
        demangled_kernel_name = demangled_kernel_name.replace("(bool)0", "false")
    else:
        call_trace_str = "::".join([s.lower() for s in temp_call_trace])
    if debug:
        print(f"\tKernel Call Trace: {kernel_call_trace}")
        print(f"\t{action.name()}")

    # Pattern ends with ":" if RAJA_CUDA, "<" if Base_CUDA
    kernel_pattern = rf"{call_trace_str}::(\w+)[<:]"
    kernel_match = re.search(kernel_pattern, demangled_kernel_name)
    # Found match
    if kernel_match:
        kernel_str = kernel_match.group(1)
        if debug:
            print(f"\tMatched {demangled_kernel_name} to kernel {kernel_str} (safe)")
    else:
        if debug:
            print(
                f"\tCould not match {demangled_kernel_name}\n\tWill still attempt to match with query from kernel call trace (unsafe)"
            )
        kernel_str = None

    # RAJA_CUDA/Lambda_CUDA variant
    instance_pattern = r"instance (\d+)"
    instance_match = re.findall(instance_pattern, demangled_kernel_name)
    if instance_match:
        instance_num = instance_match[-1]
        instance_exists = True
    else:
        # Base_CUDA variant
        instance_num = None
        instance_exists = False

    return kernel_str, demangled_kernel_name, instance_num, instance_exists, False


def _match_kernel_str_to_cali(node_set, kernel_str, instance_num, instance_exists):
    """Given a set of nodes, node_set, from querying the Caliper call
    tree using the NCU call trace, match the kernel_str to one of the
    node names. Additionally, use the instance number, instance_num to
    match kernels with multiple instances, if applicable.

    Arguments:
        node_set (list): List of Hatchet nodes from querying the call tree
        kernel_str (str): Kernel name from _match_call_trace_regex
        instance_num (int): Instance number of kernel, if applicable
        instance_exists (bool): True if instance number exists, False if not
    """
    if kernel_str:
        return [
            n
            for n in node_set
            if kernel_str in n.frame["name"]
            and (f"#{instance_num}" in n.frame["name"] if instance_exists else True)
        ]
    else:
        return [n for n in node_set if n.frame["type"] == "kernel"]


def _multi_match_fallback_similarity(matched_nodes, demangled_kernel_name, debug):
    """If _match_kernel_str_to_cali has more than one match, attempt to match using sequence similarity.

    Arguments:
        matched_nodes (list): List of matched Hatchet nodes
        demangled_kernel_name (str): Demangled kernel name from _match_call_trace_regex
        debug (bool): Print debug statements

    Returns:
        matched_node (Hatchet.node): Hatchet node with highest similarity score
    """
    # Attempt to match using similarity
    match_dict = {}
    for node in matched_nodes:
        match_ratio = SequenceMatcher(
            None, node.frame["name"], demangled_kernel_name
        ).ratio()
        match_dict[match_ratio] = node
    # Get highest ratio
    highest_ratio = max(list(match_dict.keys()))
    matched_node = match_dict[highest_ratio]
    if debug:
        print(
            f"NOTICE: Multiple matches ({len(matched_nodes)}) found for kernel. Matching using string similarity..."
        )
    return matched_node


def _build_query_from_ncu_trace(kernel_call_trace, debug):
    """Build QueryLanguage query from an NCU kernel call trace

    Arguments:
        kernel_call_trace (list): Call trace as seen from NCU
    """

    def _predicate_builder(kernel, is_regex=False):
        """Build predicate for QueryMatcher while forcing memoization

        Arguments:
            kernel (str): kernel name
            is_regex (bool): whether kernel is a regex

        Returns:
            predicate (function): predicate function
        """
        if is_regex:
            return (
                lambda row: row["name"]
                .apply(lambda x: kernel in x if isinstance(x, str) else False)
                .all()
            )
        else:
            return lambda row: row["name"].apply(lambda x: x == kernel).all()

    query = QueryMatcher()
    for i, kernel in enumerate(kernel_call_trace):
        if i == 0:
            query.match(".", _predicate_builder(kernel))
        elif i == len(kernel_call_trace) - 1:
            query.rel("*")
            query.rel(".", _predicate_builder(kernel, is_regex=True)).rel("*")
        else:
            query.rel(".", _predicate_builder(kernel))

    return query


class NCUReader:
    """Object to interface and pull NCU report data into Thicket"""

    def _read_ncu(self, thicket, ncu_report_mapping, debug=False, disable_tqdm=False):
        """Read NCU report files and return dictionary of data.

        Arguments:
            thicket (Thicket): thicket object to add ncu metrics to
            ncu_report_mapping (dict): mapping from NCU report file to profile
            debug (bool): whether to print debug statements
            disable_tqdm (bool): whether to disable tqdm progress bar

        Returns:
            data_dict (dict): dictionary of NCU data where key is tuple, (node, profile), mapping to list of dictionaries for per-rep data that is aggregated down to one dictionary.
        """
        # Lazy import ncu_report
        import ncu_report

        # Rollup operations
        self.rollup_operations = {
            None: None,
            ncu_report.IMetric.RollupOperation_AVG: pd.Series.mean,  # 1
            ncu_report.IMetric.RollupOperation_MAX: pd.Series.max,  # 2
            ncu_report.IMetric.RollupOperation_MIN: pd.Series.min,  # 3
            ncu_report.IMetric.RollupOperation_SUM: pd.Series.sum,  # 4
        }

        # Initialize dict
        data_dict = defaultdict(list)
        rollup_dict = {}
        # Kernel mapping from NCU kernel to thicket node to save re-querying
        kernel_map = {}
        # Query results keyed by normalized NCU call trace so the same thicket
        # tree query is only executed once.
        query_cache = {}
        profile_mapping_flipped = {v: k for k, v in thicket.profile_mapping.items()}

        # Loop through NCU files
        for ncu_report_file in ncu_report_mapping:
            # Set error check flag
            call_trace_found = False

            # NCU hash
            ncu_hash = profile_mapping_flipped[ncu_report_mapping[ncu_report_file]]

            # Load file
            report = ncu_report.load_report(ncu_report_file)

            # Error check
            num_ranges = report.num_ranges()
            if num_ranges > 1:
                raise ValueError(
                    "NCU report file "
                    + ncu_report_file
                    + " has multiple ranges. Not supported yet."
                )
            elif num_ranges == 0:
                raise ValueError(
                    "NCU report file " + ncu_report_file + " has no ranges (no data)."
                )
            # Loop through ranges in report
            for range in report:
                # Grab first action
                first_action = range.action_by_idx(0)
                # Metric names
                metric_names = [
                    first_action[name].name() for name in first_action.metric_names()
                ]
                # Setup rollup dict
                rollup_dict = {
                    name: first_action[name].rollup_operation() for name in metric_names
                }

                # Query action in range
                pbar = tqdm(range, disable=disable_tqdm)
                for i, action in enumerate(pbar):
                    pbar.set_description(f"Processing action {i}/{len(range)}")
                    if debug:
                        print(f"Action: {i}")
                    # Demangled name of kernel
                    demangled_kernel_name = action.name(
                        ncu_report.IAction.NameBase_DEMANGLED
                    )
                    # Get NCU-side kernel trace
                    kernel_call_trace = list(
                        action.nvtx_state().domain_by_id(0).push_pop_ranges()
                    )

                    # Skip warmup kernels
                    if len(kernel_call_trace) == 0:
                        continue
                    else:
                        call_trace_found = True

                        (
                            kernel_str,
                            demangled_kernel_name,
                            instance_num,
                            instance_exists,
                            skip_kernel,
                        ) = _match_call_trace_regex(
                            kernel_call_trace, demangled_kernel_name, debug, action
                        )
                        if skip_kernel:
                            continue

                        # Add kernel name to the end of the trace tuple
                        if kernel_str:
                            kernel_call_trace.append(kernel_str)

                        # Match ncu kernel to thicket node
                        matched_node = None
                        if demangled_kernel_name in kernel_map and kernel_str:
                            # Skip query building
                            matched_node = kernel_map[demangled_kernel_name]
                            if debug:
                                print(
                                    f"\tKernel already in mapping: {demangled_kernel_name}"
                                )
                        else:  # kernel hasn't been seen yet
                            query_key = tuple(kernel_call_trace)
                            if query_key in query_cache:
                                node_set = query_cache[query_key]
                                if debug:
                                    print(f"\tQuery already in mapping: {query_key}")
                            else:
                                # Build and apply the query once per unique trace.
                                query = _build_query_from_ncu_trace(
                                    kernel_call_trace, debug
                                )
                                node_set = query.apply(thicket)
                                query_cache[query_key] = node_set
                            # Find the correct node. This may also get the parent so we take the last one
                            matched_nodes = _match_kernel_str_to_cali(
                                node_set,
                                kernel_str,
                                instance_num,
                                instance_exists,
                            )
                            if len(matched_nodes) > 1:
                                matched_node = _multi_match_fallback_similarity(
                                    matched_nodes, demangled_kernel_name, debug
                                )
                            elif len(matched_nodes) == 1:
                                matched_node = matched_nodes[0]
                            else:
                                raise ValueError(
                                    "No node found for kernel: " + kernel_str
                                )

                            if debug:
                                if not instance_exists:
                                    instance_num = "NA"
                                print(
                                    f"\tMatched NCU kernel:\n\t\t{demangled_kernel_name}\n\tto Caliper Node:\n\t\t{matched_node}"
                                )
                                print(
                                    f"\tAKA:\n\t\t{kernel_str} (instance {instance_num}) == {kernel_str} (#{instance_num})\n"
                                )
                                print("\tAll matched nodes:")
                                for node in matched_nodes:
                                    print("\t", node)

                        # Set mapping
                        kernel_map[demangled_kernel_name] = matched_node

                        metric_values = [action[name].value() for name in metric_names]

                        assert len(metric_names) == len(metric_values)
                        data_dict[(matched_node, ncu_hash)].append(
                            dict(zip(metric_names, metric_values))
                        )

            if not call_trace_found:
                raise ValueError(
                    f"No kernel call traces found in {ncu_report_file}.\nCheck you are enabling the NVTX Caliper service when running NCU."
                )

        return data_dict, rollup_dict
