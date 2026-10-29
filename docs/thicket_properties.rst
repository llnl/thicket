..
   Copyright 2022 Lawrence Livermore National Security, LLC and other
   Thicket Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

.. _thicket-properties:

#####################################################
 Thicket Object Attributes and Enforceable Properties
#####################################################

This page describes the attributes of a ``Thicket`` object, the consistency
requirements for its nodes and profiles, and the functions that validate those
requirements. Use it when implementing or modifying Thicket methods to identify
which attributes must remain synchronized and which validators apply.

A ``Thicket`` object is an instance of :class:`~thicket.thicket.Thicket`,
which subclasses ``hatchet.GraphFrame``. It has the following instance attributes:

1. ``graph`` references a ``hatchet.Graph`` of ``hatchet.Node`` objects.
2. ``dataframe`` references a ``pandas.DataFrame`` containing performance
   measurements.
3. ``metadata`` references a ``pandas.DataFrame`` containing profile metadata.
4. ``statsframe`` references a ``hatchet.GraphFrame`` whose ``dataframe`` stores
   statistics computed across profiles.

Thicket methods require specific *invariants* across these attributes. For
example, ``statsframe.graph is graph`` must hold, and the ``node`` index levels
of ``dataframe`` and ``statsframe.dataframe`` must reference the same
``hatchet.Node`` instances as ``graph``. Profile IDs in the DataFrame indexes
must correspond to entries in ``profile`` and keys in ``profile_mapping``,
subject to the column-wise concatenation rules described below.

*********************
 Instance Attributes
*********************

.. list-table::
   :header-rows: 1
   :class: wrap
   :widths: 24 46 30

   * - Attribute
     - Contents
     - Index
   * - ``graph``
     - A ``hatchet.Graph`` of ``hatchet.Node`` objects, one for each call tree
       node of any profile
     - —
   * - ``dataframe``
     - A ``pandas.DataFrame`` of performance measurements, with metric columns
       and a ``name`` column; one row per unique combination of index values
     - ``node``, then the profile level, then any levels added by the reader,
       such as ``rank``
   * - ``metadata``
     - A ``pandas.DataFrame`` with one row per profile
     - The profile level
   * - ``statsframe``
     - A ``hatchet.GraphFrame`` whose ``dataframe`` stores aggregated statistics
       and whose ``graph`` references the same object as this Thicket's ``graph``
     - ``statsframe.dataframe`` is indexed by ``node``
   * - ``profile``
     - A ``list`` of profile IDs
     - —
   * - ``profile_mapping``
     - A ``collections.OrderedDict`` mapping each profile ID to its source,
       usually a file path
     - —
   * - ``profile_idx_name``
     - A ``str`` naming the profile index level, ``"profile"`` by default
     - —

Profile IDs
===========

A *profile ID* identifies a profile in ``dataframe``, ``metadata``, ``profile``,
and ``profile_mapping``. By default, it is an integer computed from the file
path by :meth:`Thicket.profile_hasher()
<thicket.thicket.Thicket.profile_hasher>`. :meth:`Thicket.replace_profile_index()
<thicket.thicket.Thicket.replace_profile_index>` replaces the IDs with the
values of a metadata column, which can be of any type but must be unique, and
renames the profile level after that column.

Node IDs
========

Every ``hatchet.Node`` has a ``_hatchet_nid``, called its *nid* on these pages:
its position in the traversal of its own graph, assigned by
``Graph.enumerate_traverse()``. ``Node.__eq__()``, ``__hash__()``, and
``__lt__()`` use only the nid, so nodes of different graphs can compare equal (see :ref:`concat-unify-graphs`).

***********************
 Enforceable Properties
***********************

.. _prop-node-identity:

Node Identity
=============

``graph`` and the ``node`` index levels of ``dataframe`` and
``statsframe.dataframe`` contain the same ``hatchet.Node`` instances, compared
by Python object identity.

:Enforced by: :func:`~thicket.utils.validate_nodes`.

.. _prop-contiguous-nids:

Contiguous Node IDs
===================

The distinct ``_hatchet_nid`` values in ``dataframe`` form a contiguous sequence
from zero to the number of distinct IDs minus one.

:Enforced by: :func:`~thicket.utils.validate_dataframe`.

.. _prop-unique-rows:

Unique Performance Rows
=======================

Each combination of index values in ``dataframe`` is unique, including the
node, profile ID, and any additional levels such as ``rank``.

:Enforced by: :func:`~thicket.utils.validate_dataframe`.

.. _prop-name-column:

Name Column
===========

Each value in the ``name`` column equals the corresponding
``node.frame["name"]``, or is ``None`` for a row inserted by
``utils._fill_perfdata()``. After column-wise concatenation, the column label
is ``("name", "")``.

:Enforced by: :func:`~thicket.utils.validate_dataframe`.

.. _prop-column-structure:

Matching Column Structure
=========================

``dataframe.columns`` and ``metadata.columns`` are either both
``pandas.MultiIndex`` instances or both single-level indexes.

:Enforced by: :func:`~thicket.utils.validate_profile`.

.. _prop-profiles-agree:

Consistent Profile IDs
======================

``dataframe``, ``metadata``, ``profile``, and ``profile_mapping`` identify the
same set of profiles. The validator compares ``dataframe.index`` with its
``node`` level removed, ``metadata.index``, ``profile``, and the keys of
``profile_mapping``. For tuple IDs created by column-wise concatenation, it
accepts matching tuple elements or combinations of elements in the table indexes.

:Enforced by: :func:`~thicket.utils.validate_profile`.

.. _prop-unique-profiles:

Unique Profiles
===============

``profile`` and ``metadata.index`` contain no duplicate profile IDs.

:Enforced by: :func:`~thicket.utils.validate_profile`.

.. _prop-sorted-profiles:

Sorted Profiles
===============

Profile IDs appear in sorted order of first occurrence in the innermost index
level of both ``dataframe`` and ``metadata``.

:Enforced by: :func:`~thicket.utils.verify_sorted_profile`.
