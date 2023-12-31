import pandas as pd

from src import DataPipeline
from src import (
    normalize_features,
    convert_units,
    remove_nighttime_values,
    plot_forecast
)
from src.models.forecasting_model import IrradianceForecastingModel
from src.utils.evaluation import summary_stats


class Model:
    """
    A class for training and evaluating different models using the provided data pipeline.

    Attributes:
        data_pipeline (DataPipeline): A preprocessed data pipeline containing the training and testing data.
    """
    def __init__(self, data_pipeline: DataPipeline, from_pretrained=False, save_models=False) -> None:
        """
        Initialize the Model object with a given DataPipeline.

        Args:
            data_pipeline (DataPipeline): A preprocessed data pipeline
                                          containing the training and testing data.
        """
        assert data_pipeline.split_data == True
        self.from_pretrained = from_pretrained
        self.dp = data_pipeline
        model = IrradianceForecastingModel(
                                            data_pipeline.target_variable,
                                            data_pipeline.time_horizon,
                                            from_pretrained = from_pretrained,
                                            save_models = save_models,
                                           )
        self.models = [
            ["nn", model.train_nn, model.forecast_nn],
            ["OLS", model.train_ols, model.forecast_ols],
        ]

    def itterate_through_data(self) -> None:
        """
        Iterate through the data and train models for each feature using the provided model.

        Returns:
            None
        """
        results = {}
        for Xtra,Xtes,f in self.dp.itterator:
            Xtra, Xtes = normalize_features(Xtra, Xtes)
            for name, train_fn, pred_fn in self.models:
                t, h = self.dp.target_variable, self.dp.time_horizon, 
                print(f, name, t, h)
                model = train_fn(Xtra, self.dp.train_y, f)
                train_pred = pred_fn(model, Xtra)
                test_pred = pred_fn(model, Xtes)
                # convert from kt [-] back to irradiance [W/m^2]
                convert_units(train_pred, self.dp.train_clear)
                convert_units(test_pred, self.dp.test_clear)
                # removes nighttime values (solar elevation < 5)
                remove_nighttime_values(train_pred, self.dp.train_clear)
                remove_nighttime_values(test_pred, self.dp.test_clear)
                # plot_forecast(self.dp.test[[f'{t}_{h}']], test_pred, name=f'test_full_model_{name}_{t}_{h}')
                # plot_forecast(self.dp.train[[f'{t}_{h}']], train_pred, name=f'train_full_model_{name}_{t}_{h}')
                plot_forecast(self.dp.test.loc[:, [f'{t}_{h}']], test_pred, name=f'test_full_model_{name}_{t}_{h}')
                plot_forecast(self.dp.train.loc[:, [f'{t}_{h}']], train_pred, name=f'train_full_model_{name}_{t}_{h}')
                self.dp.train.insert(self.dp.train.shape[1], "{}_{}_{}".format(self.dp.target, name,f), train_pred)
                self.dp.test.insert(self.dp.test.shape[1], "{}_{}_{}".format(self.dp.target, name,f), test_pred)
                stats = summary_stats(self.dp.test, test_pred, self.dp, name, f)

                for k,v in stats.items():
                    if k not in results:
                        results[k]=[v]
                    else:
                        results[k].append(v)
        return pd.DataFrame(results)