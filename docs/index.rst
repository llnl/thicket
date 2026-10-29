..
   Copyright 2022 Lawrence Livermore National Security, LLC and other
   Thicket Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

..
   thicket documentation master file, created by
   sphinx-quickstart on Tue Jun 26 08:43:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#########
 Thicket
#########

Thicket is a python-based toolkit for Exploratory Data Analysis (EDA) of parallel
performance data that enables performance optimization and understanding of
applications' performance on supercomputers. It bridges the performance tool gap between
being able to consider only a single instance of a simulation run (e.g., single
platform, single measurement tool, or single scale) and finding actionable insights in
multi-dimensional, multi-scale, multi-architecture, and multi-tool performance datasets.

You can get thicket from its `GitHub repository <https://github.com/llnl/thicket>`_:

.. code:: console

   $ git clone https://github.com/llnl/thicket.git

or install it using pip:

.. code:: console

   $ pip install llnl-thicket

If you are new to thicket and want to start using it, see :doc:`Getting Started
<getting_started>`.

.. toctree::
   :maxdepth: 2
   :caption: User Docs

   getting_started
   user_guide
   generating_data

If you encounter bugs while using thicket, you can report them by opening an issue on
`GitHub <http://github.com/llnl/thicket/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial_materials
   thicket_tutorial.ipynb
   thicket_rajaperf_clustering.ipynb
   extrap-with-metadata-aggregated.ipynb
   stats-functions.ipynb
   query_language.ipynb
   groupby_aggregate.ipynb
   nsight_compute.ipynb
   composing_parallel_sorting_data.ipynb
   modeling_parallel_sorting_data.ipynb
   tma_speedup_clustering.ipynb
   vis_docs

.. toctree::
   :maxdepth: 2
   :caption: Reference

   publications

.. toctree::
   :maxdepth: 2
   :caption: Developer Docs

   developer_guide
   thicket_properties
   concatenating_thickets

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   Thicket API Docs <source/thicket>

####################
 Indices and tables
####################

-  :ref:`genindex`
-  :ref:`modindex`
-  :ref:`search`
