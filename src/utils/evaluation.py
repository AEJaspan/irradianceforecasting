import pandas as pd
import numpy as np

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
    for filename in filenames:
        df = pd.read_hdf(filename, "df")

        # essential metadata
        horizon = df["horizon"].values[0]
        #print(df.describe())

        # error metrics
        for dataset, group in df.groupby("dataset"):
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

    # return as a DataFrame
    return pd.DataFrame(results)