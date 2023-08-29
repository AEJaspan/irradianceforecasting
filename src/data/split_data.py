from root import ROOT_DIR
import pandas as pd
import src.features


class DataPipeline:
    """A class to manage loading and splitting of data for forecasting."""
    def __init__(self, dev_limit=1) -> None:
        """Initialize the DataPipeline object."""
        self.data_loaded = False
        self.split_data = False
        self.dev_limit = dev_limit
    
    def load_data(self):
        """
        Load endogenous, exogenous, baseline, and target data into the DataPipeline.
        """
        self.endogenous = pd.read_csv(ROOT_DIR/'data/raw/Irradiance_features_intra-hour.csv', 
                                      parse_dates=True,  index_col='timestamp')[::self.dev_limit]
        self.exogenous  = pd.read_csv(ROOT_DIR/'data/raw/Sky_image_features_intra-hour.csv',
                                      parse_dates=True, index_col='timestamp')[::self.dev_limit]
        self.target     = pd.read_csv(ROOT_DIR/'data/raw/Target_intra-hour.csv',
                                      parse_dates=True, index_col='timestamp')[::self.dev_limit]
        self.baseline   = pd.read_csv(ROOT_DIR/'data/external/baseline_intra-hour.csv',
                                      parse_dates=True, index_col='timestamp')[::self.dev_limit]
        self.data_loaded=True
    
    def train_test_split(self, target_variable, time_horizon):
        """
        `target_variable`  stores the string defining the target variable for the forecast.
                It is set to either "ghi" or "dni", which represent different measures of solar irradiance.
        `time_horizon` represents the time interval for which the forecast is being made.
                It is used to specify the different forecast horizons for which the model will be run.
                The code iterates over the `horizon` list and performs the forecast for each specified horizon.
        returns:
                instance of DataPipeline
        """
        assert self.data_loaded == True
        inpEndo = self.endogenous
        inpExo  = self.exogenous
        tar     = self.target
        self.target_variable = target_variable
        self.time_horizon = time_horizon
        cols = [
                "{}_{}".format(target_variable,time_horizon),  # actual
                "{}_kt_{}".format(target_variable,time_horizon),  # clear-sky index
                "{}_clear_{}".format(target_variable,time_horizon),  # clear-sky model
                "elevation_{}".format(time_horizon)   # solar elevation 
            ]

        train = inpEndo[inpEndo.index.year <= 2015]
        train = train.join(inpExo[inpEndo.index.year <= 2015], how="inner")
        train = train.join(tar[tar.index.year <= 2015], how="inner")

        test = inpEndo[inpEndo.index.year == 2016]
        test = test.join(inpExo[inpEndo.index.year == 2016], how="inner")
        test = test.join(tar[tar.index.year == 2016], how="inner")

        feature_cols = inpEndo.filter(regex=target_variable).columns.tolist()
        feature_cols_endo = inpEndo.filter(regex=target_variable).columns.tolist()
        feature_cols.extend(inpExo.columns.unique().tolist())
            
        train = train[cols + feature_cols].dropna(how="any")
        test  = test[cols + feature_cols].dropna(how="any")

        train_X = train[feature_cols].values
        test_X  = test[feature_cols].values
        train_X_endo = train[feature_cols_endo].values
        test_X_endo  = test[feature_cols_endo].values

        self.train_y = train["{}_kt_{}".format(target_variable,time_horizon)].values
        elev_train = train["elevation_{}".format(time_horizon)].values
        elev_test  = test["elevation_{}".format(time_horizon)].values

        self.train_clear = train["{}_clear_{}".format(target_variable,time_horizon)].values
        self.test_clear = test["{}_clear_{}".format(target_variable,time_horizon)].values
        self.train = train
        self.test = test
        self.itterator = zip([train_X_endo,train_X],[test_X_endo,test_X],['endo','exo'])
        self.split_data=True
        return self