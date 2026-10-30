..
   Copyright 2022 Lawrence Livermore National Security, LLC and other
   Thicket Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

#################
 Getting Started
#################

***************
 Prerequisites
***************

Thicket has the following minimum requirements, which must be installed before Thicket
is run:

#. Python 3 (3.11 or newer)
#. hatchet
#. pandas >= 3.0
#. numpy
#. matplotlib, and
#. scipy

****************
 Other Packages
****************

#. Extrap: extrap, matplotlib
#. Vis: beautifulsoup4
#. Plotting: seaborn

For installation options for the extra packages, refer to the installation instructions
below in `Install and Build Thicket`_. Thicket is available on `GitHub
<https://github.com/llnl/thicket>`_.

**************
 Installation
**************

You can get thicket from its `GitHub repository <https://github.com/llnl/thicket>`_
using this command:

.. code:: console

   $ git clone https://github.com/llnl/thicket.git

This will create a directory called ``thicket``.

Install and Build Thicket
=========================

To build thicket and update your PYTHONPATH, run the following shell script from the
thicket root directory:

.. code:: console

   $ source ./install.sh

Note: The ``source`` keyword is required to update your PYTHONPATH environment variable.
It is not necessary if you have already manually added the thicket directory to your
PYTHONPATH.

Alternatively, you can install thicket using pip:

.. code:: console

   $ pip install llnl-thicket

You can install the other packages mentioned above for additional features of thicket.
Below is an example of installing thicket with extrap.

.. code:: console

   $ pip install llnl-thicket[extrap]

Check Installation
==================

After installing thicket, you should be able to import thicket when running the Python
interpreter in interactive mode:

.. code:: console

   $ python
   Python 3.7.4 (default, Jul 11 2019, 01:08:00)
   [Clang 10.0.1 (clang-1001.0.46.4)] on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   >>>

Typing ``import thicket`` at the prompt should succeed without any error messages:

.. code:: console

   >>> import thicket
   >>>

Interactive Visualization
=========================

Thicket provides an interactive visualization which can be run inside of your Jupyter
notebooks. It is dependent on different mechanism for building, which we describe here.

The software in the ``thicket/vis`` subdirectory (i.e., the ``thicket.vis`` package)
requires `Node.js and the Node Package Manager (NPM) <https://nodejs.org/en/download/>`_
for the development and building of JavaScript code.

Building Visualization Code for Users
=====================================

If you are just using our built-in visualizations, the visualization code will be built
automatically when you access the ``thicket.vis`` module. All that users have to do is
make sure they have NPM installed. If NPM is not installed, accessing the ``thicket.vis``
module will raise a ``FileNotFoundError``.

Building Visualization Code for Developers
==========================================

If you are developing a visualization, it is recommended that you build the visualization
code manually. To manually build this code, follow the instructions below.

Installing Node Packages
========================

Once you have Node and NPM installed on your system, you can install all necessary node
packages by running the following line in your terminal from the ``thicket/vis``
directory:

.. code:: console

   >>> npm install

Building Out JavaScript Code with Webpack
=========================================

To build out JavaScript into the static bundles used by the Jupyter visualizations, run
the following line from the ``thicket/vis`` directory in your terminal:

.. code:: console

   >>> npm run build

Alternatively, you can run the following line to force bundles to automatically update
when you change the JavaScript source code:

.. code:: console

   >>> npm run watch
