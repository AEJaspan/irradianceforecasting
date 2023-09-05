"""Console script for pollscraper."""
import sys
import click
import pandas as pd

from src import DataPipeline
from src import Model
from src import plot_skill


@click.command()
@click.option('--from_pretrained', default=False, is_flag=True,
              help='Option to load pretrained models stored in the '
              '<${ROOT_DIR}/models> directory [defaults to False]')
@click.option('--save_models', default=False, is_flag=True,
              help='Option to save new models to the '
              '<${ROOT_DIR}/models> directory [defaults to False]')
def main(from_pretrained, save_models):
    """
    Main entry point for the IrradianceForecasting pipeline.
    
    Call this from the command line with the command:
    
    `$ IrradianceForecasting`
    
    """
    dp = DataPipeline(dev_limit=1)
    dp.load_data()
    targets  = ["dni", "ghi"]
    horizons = ["5min", "10min", "15min", "20min", "25min", "30min"]
    results=pd.DataFrame()
    for t in targets:
        h = horizons[-1]
        target, horizon = t, h
        dp.train_test_split(target, horizon)
        model = Model(dp, from_pretrained=from_pretrained,
                      save_models=save_models)
        update = model.itterate_through_data()
        results = pd.concat([results,update])
    results.reset_index(drop=True, inplace=True)
    plot_skill(results)
    print(f"IrradianceForecasting has successfully completed! "
          f"Please find all figures in the <${ROOT_DIR}/reports/figures/> "
          f"directory, and all models in the <${ROOT_DIR}/models>"
          "directory.")
    return 0


if __name__ == "__main__":
    cmd = '--from_pretrained'
    print(f"Running $ IrradianceForecasting with options: {cmd}")
    main(cmd.split())
#    sys.exit(main())
