Data Dictionary
===============

Units
======
All time stamps are in UTC (YYYY-MM-DD HH:MM:SS).
All irradiance and weather data are in SI units.
Sky image features are derived from 8-bit RGB (256 color levels) data.
Satellite images are derived from 8-bit gray-scale (256 color levels) data.

Missing data
============
The string "NAN" indicates missing data

File formats
============
All time series data files as in CSV (comma separated values)
Images are given in tar.bz2 files

Data Description
==================

* Day ahead weather forecasts for closest weather stations
   * Granularity: Hourly
* Satellite imagery from GOES-15
   * Granularity: 15 Minute
* Imagery from Whole Sky Cameras
   * Granularity: Minutely

Models are developed on historical data including 2014 and 2015, and are then evaluated against 2015 data.

The primary datasets for solar forecasting are the two main modes of solar irradiance, namely, GHI (global) and DNI (beam). These two variables are used to train the models and assess the fore- casting performance.
DNI is computed directly from the GHI, DHI, and solar zenith angle (:math:`theta_{z}`).

Tables
======

.. csv-table:: List of extracted NAM variables.
   :file: _static/NAM-Variables.csv
   :header-rows: 1

.. csv-table:: Nomenclature.
   :file: _static/Nomenclature.csv
   :header-rows: 1

.. csv-table:: intra times.
   :file: _static/intra-times.csv
   :delim: ;
   :header-rows: 1

.. csv-table:: NAM Gridpoints.
   :file: _static/NAM-Gridpoints.csv
   :header-rows: 1


.. csv-table:: File Dictionary
   :file: _static/file-dictionary.csv
   :header-rows: 1
   :delim: :


List of files available in the data repository. The second column indicates the type of information contained in the file, where “Primary” refers to quality controlled
data from the primary sources (solar sensor, sky imager, and NWP), “Secondary” refers to data obtained by processing the primary data, and “Code” refers to Python 3 scripts.
All these files can be accessed at https://doi.org/10.5281/zenodo.2826939.