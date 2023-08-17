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

Tables
======

.. csv-table:: List of extracted NAM variables.
   :file: NAM-Variables.csv
   :header-rows: 1

.. csv-table:: Nomenclature.
   :file: Nomenclature.csv
   :header-rows: 1

.. csv-table:: intra times.
   :file: intra-times.csv
   :delim: ;
   :header-rows: 1

.. csv-table:: NAM Gridpoints.
   :file: NAM-Gridpoints.csv
   :header-rows: 1


.. csv-table:: File Dictionary
   :file: file-dictionary.csv
   :header-rows: 1
   :delim: :


List of files available in the data repository. The second column indicates the type of information contained in the file, where “Primary” refers to quality controlled
data from the primary sources (solar sensor, sky imager, and NWP), “Secondary” refers to data obtained by processing the primary data, and “Code” refers to Python 3 scripts.
All these files can be accessed at https://doi.org/10.5281/zenodo.2826939.