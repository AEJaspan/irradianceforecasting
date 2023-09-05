[![Documentation Build Status](https://readthedocs.org/projects/irradianceforecasting/badge/?version=latest)](https://irradianceforecasting.readthedocs.io/en/latest/?version=latest)


# IrradianceForecasting
==============================

The goal is to develop a solution for short-term irradiance forecasting using historical data. Accurate irradiance forecasting is essential for renewable energy generators, especially solar power plants, to optimize their energy production. Data collected from Zenodo [id: [2826939](https://zenodo.org/record/2826939)] \cite{carreira_pedro_hugo_2019_2826939}.

Full documentation of this task can be found on [Read the docs](https://readthedocs.org/projects/irradianceforecasting/).

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Optional installations:
pip install -r optional-requirements.txt
"
```

## Run forecasting

* To train the models and produce irradiance forecasts:

```
$ IrradianceForecasting
```

*** Note: *** This command will not save the trained models. To persist the trained models to the `models/` directory, use the flag `--save_models`.

* To produce irradiance forecasts from pre-trained models:

```
$ IrradianceForecasting --from_pretrained
```

## Collect data

```
source data/raw/collect.sh
```

**NOTE** Python scripts downloaded from Zenodo require minor modifications. It is recommended to use the scripts provided in this repository.

## Building documentation

```
cd docs
make html
```


## Testing and Linting

```
tox
```


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── cli.py         <- Main entry point
    │   │
    │   ├── data           <- Scripts to process data
    │   │   └── split_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └─── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       └─── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io




## Credits

This package was created with Cookiecutter and the [`cookiecutter-data-science`](https://github.com/drivendata/cookiecutter-data-science) project template.
