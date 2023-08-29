"""Console script for pollscraper."""
import sys
import click
import pandas as pd

from src import DataPipeline
from src import Model
from src import plot_skill


@click.command()
def main():
    """
    Main entery point for the IrradianceForecasting pipeline
    """
    dp = DataPipeline(dev_limit=1000)
    dp.load_data()
    targets  = ["dni", "ghi"]
    horizons = ["5min", "10min", "15min", "20min", "25min", "30min"]
    results=pd.DataFrame()
    for t in targets:
        h = horizons[-1]
        target, horizon = t, h
        dp.train_test_split(target, horizon)
        model = Model(dp)
        results = pd.concat([results,model.itterate_through_data()])
    plot_skill(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
