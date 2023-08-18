from pathlib import Path
from root import ROOT_DIR, Path
import pandas as pd
import src.features

class DataPipeline:
    def __init__(self) -> None:
        self.data_loaded=False
        self.split_data=False
        # # , *args, **kwargs): super(CLASS_NAME, self).__init__(*args, **kwargs)
    
    def load_data(self):
        self.endogenous = pd.read_csv(ROOT_DIR/'data/raw/Irradiance_features_intra-hour.csv', parse_dates=True,  index_col='timestamp')
        self.exogenous  = pd.read_csv(ROOT_DIR/'data/raw/Sky_image_features_intra-hour.csv', parse_dates=True, index_col='timestamp')
        self.target     = pd.read_csv(ROOT_DIR/'data/raw/Target_intra-hour.csv', parse_dates=True, index_col='timestamp')
        self.data_loaded=True
    
    def train_test_split(self, target, horizon):
        """
        `target`  stores the string defining the target variable for the forecast.
                It is set to either "ghi" or "dni", which represent different measures of solar irradiance.
        `horizon` represents the time interval for which the forecast is being made.
                It is used to specify the different forecast horizons for which the model will be run.
                The code iterates over the `horizon` list and performs the forecast for each specified horizon.
        returns:
                Zipped itterable in the order:
                    [training_features, testing_features, column_string['endo'/'exo']]
        """
        assert self.data_loaded == True
        inpEndo = self.endogenous
        inpExo  = self.exogenous
        tar     = self.target
        cols = [
                "{}_{}".format(target,horizon),  # actual
                "{}_kt_{}".format(target,horizon),  # clear-sky index
                "{}_clear_{}".format(target,horizon),  # clear-sky model
                "elevation_{}".format(horizon)   # solar elevation 
            ]

        train = inpEndo[inpEndo.index.year <= 2015]
        train = train.join(inpExo[inpEndo.index.year <= 2015], how="inner")
        train = train.join(tar[tar.index.year <= 2015], how="inner")

        test = inpEndo[inpEndo.index.year == 2016]
        test = test.join(inpExo[inpEndo.index.year == 2016], how="inner")
        test = test.join(tar[tar.index.year == 2016], how="inner")

        feature_cols = inpEndo.filter(regex=target).columns.tolist()
        feature_cols_endo = inpEndo.filter(regex=target).columns.tolist()
        feature_cols.extend(inpExo.columns.unique().tolist())
            
        train = train[cols + feature_cols].dropna(how="any")
        test  = test[cols + feature_cols].dropna(how="any")

        train_X = train[feature_cols].values
        test_X  = test[feature_cols].values
        train_X_endo = train[feature_cols_endo].values
        test_X_endo  = test[feature_cols_endo].values

        train_y = train["{}_kt_{}".format(target,horizon)].values
        elev_train = train["elevation_{}".format(horizon)].values
        elev_test  = test["elevation_{}".format(horizon)].values

        train_clear = train["{}_clear_{}".format(target,horizon)].values
        test_clear = test["{}_clear_{}".format(target,horizon)].values
        self.itterator = zip([train_X_endo,train_X],[test_X_endo,test_X],['endo','exo'])
        self.split_data=True
        return self