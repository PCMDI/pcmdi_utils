#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    # TODO: Update author and author_email
    author="Jiwoo Lee",
    author_email="lee1043@llnl.gov",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # TODO: Update description as needed
    description="Python Template Repo contains all the boilerplate you need to create a Python package.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    # TODO: Update keywords and name to your library name
    keywords="pcmdi_utils",
    name="pcmdi_utils",
    # TODO: Update include statement in packages to your library modules
    packages=find_packages(include=["pcmdi_utils", "pcmdi_utils.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    data_files=[("share/pcmdi_utils", ["data/navy_land.nc"])],
    # TODO: Update url to the library's GitHub URL
    url="https://github.com/PCMDI/pcmdi_utils",
    version="0.1.0",
    zip_safe=False,
)
