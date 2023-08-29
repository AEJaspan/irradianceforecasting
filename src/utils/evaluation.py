import numpy as np

def summary_stats(test, pred, dp, model_str, f, baseline_str='sp'):
    """Utility function to calculate the summary statistics
    """    
    # error metrics [W/m^2]
    error = pred - test[f"{dp.target_variable}_{dp.time_horizon}"]
    mae = np.nanmean(np.abs(error))
    mbe = np.nanmean(error)
    rmse = np.sqrt(np.nanmean(error ** 2))
    baseline = dp.baseline[f"Test_ghi_{baseline_str}_{dp.time_horizon}"]
    # forecast skill [-]:
    #
    #       s = 1 - RMSE_f / RMSE_p
    #
    # where RMSE_f and RMSE_p are the RMSE of the forecast model
    # and reference baseline model, respectively.
    rmse_p = np.sqrt(
        np.mean((pred - baseline) ** 2)
    )
    skill = 1.0 - rmse / rmse_p
    return {
                "type": f,
                "target": dp.target_variable,
                "horizon": dp.time_horizon,
                "model": model_str,
                "MAE": mae,
                "MBE": mbe,
                "RMSE": rmse,
                "skill": skill,
                "baseline": baseline_str,
            }

