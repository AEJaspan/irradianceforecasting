.. IrradianceForecasting documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

IrradianceForecasting documentation!
==============================================

Intro:
===========

The goal is to develop a solution for short-term irradiance forecasting using historical data. Accurate irradiance forecasting is essential for renewable energy generators, especially solar power plants, to optimize their energy production. Data collected from Zenodo [id: 2826939] \cite{carreira_pedro_hugo_2019_2826939}.

The approach taken in this project is to compare directly an ordinary least squares (OLS) linear regression model against a small, 4 layer neural network (nn) with 1937 parameters. This package is constructed in a modular way, and is designed to be a framework for developing and implementing more complex models. Alternative methods, beyond a neural network had been developed, but were dropped in the interest of time, and to keep the core framework clean.

Contents:
===========

.. toctree::
   :maxdepth: 3

   getting-started
   usage
   data-dictionary
   source/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
