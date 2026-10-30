# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from setuptools import setup
from codecs import open
from os import path


def readme():
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()


# Get the version in a safe way which does not reference thicket `__init__` file
# per python docs: # https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open("./thicket/version.py") as fp:
    exec(fp.read(), version)


setup(
    name="llnl-thicket",
    version=version["__version__"],
    license="MIT",
    description="Toolkit for exploratory data analysis of ensemble performance data",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords="",
    project_urls={
        "Source Code": "https://github.com/LLNL/thicket",
        "Documentation": "https://thicket.readthedocs.io/",
    },
    python_requires=">=3.11",
    packages=[
        "thicket",
        "thicket.stats",
        "thicket.vis",
        "thicket.external",
    ],
    include_package_data=True,
    install_requires=[
        "scipy",
        "numpy",
        "pandas >= 3",
        # pandas 3 support is on hatchet develop (LLNL/hatchet#185). The 2026.2.0
        # release itself requires pandas<3, so this floor admits hatchet develop and
        # later releases while excluding older releases that lack a pandas pin.
        "llnl-hatchet >= 2026.2.0",
        "tqdm",
        "more-itertools",
    ],
    extras_require={
        "extrap": ["extrap", "matplotlib"],
        "plotting": ["seaborn"],
        "vis": ["beautifulsoup4"],
    },
)
