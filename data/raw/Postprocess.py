
"""Postprocess forecasts (statistics, plots, etc.)"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

processed_data_path="data/processed"


def summary_stats(target, filenames, baseline="sp"):
    """Compute summary statistics (MAE, MBE, etc.).

    Parameters
    ----------
    target : str {"ghi", "dni"}
        Target variable.
    filenames : list
        List of filenames (relative or absolute) containing forecast time-series.
    baseline : str
        The baseline forecast to compare against.

    Returns
    -------
    df : pandas.DataFrame
        Summary statistics (MAE, MBE, etc.) by horizon, dataset, feature set,
        and model.

    """

    results = []
    merge=pd.DataFrame()
    for filename in filenames:
        df = pd.read_hdf(filename, "df")

        # essential metadata
        horizon = df["horizon"].values[0]
        #print(df.describe())

        # error metrics
        for dataset, group in df.groupby("dataset"):
            bl = group["{}_{}".format(target, baseline)].copy()
            col_name=f"{dataset}_{target}_{baseline}_{horizon}"
            print(col_name)
            merge[col_name]=bl
            for model in [baseline,"ols_endo", "ridge_endo", "lasso_endo","ols_exo", "ridge_exo", "lasso_exo"]:
        
                # error metrics [W/m^2]
                error = (group["{}_{}".format(target, horizon)] - group["{}_{}".format(target, model)]).values
                mae = np.nanmean(np.abs(error))
                mbe = np.nanmean(error)
                rmse = np.sqrt(np.nanmean(error ** 2))

                # forecast skill [-]:
                #
                #       s = 1 - RMSE_f / RMSE_p
                #
                # where RMSE_f and RMSE_p are the RMSE of the forecast model
                # and reference baseline model, respectively.
                rmse_p = np.sqrt(
                    np.mean((group["{}_{}".format(target, horizon)] - group["{}_{}".format(target, baseline)]) ** 2)
                )
                skill = 1.0 - rmse / rmse_p

                results.append(
                    {
                        "dataset": dataset,  # Train/Test
                        "horizon": horizon,  # 5min, 10min, etc.
                        "model": model,
                        "MAE": mae,
                        "MBE": mbe,
                        "RMSE": rmse,
                        "skill": skill,
                        "baseline": baseline,  # the baseline forecast name
                    }
                )
        name = Path(filename).stem.split('_')[1]
        merge.to_csv(f'data/external/baseline_{name}.csv')
    # return as a DataFrame
    return pd.DataFrame(results)


def summarize(target):
    """Summarize the forecasts, per horizon range."""

    # intra-hour: 5-30min ahead
    df = summary_stats(
        target,
        glob.glob(
            os.path.join(
                processed_data_path, "forecasts_intra-hour*{}.h5".format(target)
            )
        ),
        baseline="sp",
    )
    df.to_hdf(os.path.join(processed_data_path, "results_{}_intra-hour.h5".format(target)), "df", mode="w")
    print("Intra-hour ({}): {}".format(target, df.shape))

    # intra-day: 30min to 3h ahead
    df = summary_stats(
        target,
        glob.glob(
            os.path.join(
                processed_data_path, "forecasts_intra-day*{}.h5".format(target)
            )
        ),
        baseline="sp",
    )
    df.to_hdf(os.path.join(processed_data_path, "results_{}_intra-day.h5".format(target)), "df", mode="w")
    print("Intra-day ({}): {}".format(target, df.shape))

    # day-ahead: >24h ahead, generated at 12Z
    df = summary_stats(
        target,
        glob.glob(
            os.path.join(
                processed_data_path, "forecasts_day-ahead*{}.h5".format(target)
            )
        ),
        baseline="nam",
    )
    
    df.to_hdf(os.path.join(processed_data_path, "results_{}_day-ahead.h5".format(target)), "df", mode="w")
    print("Day-ahead ({}): {}".format(target, df.shape))


def summary_table(target, horizon_set="intra-hour"):
    """Summary table for paper."""

    # results
    df = pd.read_hdf(os.path.join(processed_data_path,"results_{}_{}.h5".format(target, horizon_set)), "df")
    df = df[df["dataset"] == "Test"]
    # generate table
    for model, group in df.groupby(["model"]):
        meta_str = "{:<9} & {:<6}".format(horizon_set, model[0])
        mae_str = "{:.1f} $\\pm$ {:.1f}".format(group["MAE"].mean(), group["MAE"].std())
        mbe_str = "{:.1f} $\\pm$ {:.1f}".format(group["MBE"].mean(), group["MBE"].std())
        rmse_str = "{:.1f} $\\pm$ {:.1f}".format(group["RMSE"].mean(), group["RMSE"].std())
        skill_str = "{:.1f} $\\pm$ {:.1f}".format(group["skill"].mean() * 100, group["skill"].std() * 100)
        print("{:<30} && {:<16} & {:<16} & {:<20} & {:<18} \\\\".format(meta_str, mae_str, mbe_str, rmse_str, skill_str))
       

# computes and prints error metrics
target = "dni"
summarize(target)
summary_table(target, horizon_set="intra-hour")
summary_table(target, horizon_set="intra-day")
summary_table(target, horizon_set="day-ahead")


target = "ghi"
summarize(target)
summary_table(target, horizon_set="intra-hour")
summary_table(target, horizon_set="intra-day")
summary_table(target, horizon_set="day-ahead")

