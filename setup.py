from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='The goal is to develop a solution for short-term irradiance forecasting using historical data. Accurate irradiance forecasting is essential for renewable energy generators, especially solar power plants, to optimize their energy production. Data collected from Zenodo [id: 2826939] \cite{carreira_pedro_hugo_2019_2826939}.',
    author='Adam Jaspan',
    license='MIT',
)
