from src import DataPipeline
from src import (
    normalize_features,
    convert_units,
    remove_nighttime_values
)
from sklearn import linear_model, ensemble, neural_network

class Model:
    def __init__(self, data_pipeline: DataPipeline) -> None:
        assert data_pipeline.split_data == True
        self.dp = data_pipeline
        self.models = [
        # Ordinary Least-Squares (OLS)
        ["ols", linear_model.LinearRegression()],
        # Ridge Regression (OLS + L2-regularizer)
        ["ridge", linear_model.RidgeCV(cv=10)],
        # Lasso (OLS + L1-regularizer)
        ["lasso", linear_model.LassoCV(cv=10, n_jobs=-1, max_iter=10000)],
    ]
    
    def train(self, model, var_train, label_train):
        """_summary_

        Args:
            model (_type_): _description_
            train (_type_): _description_
        """
        model.fit(var_train, label_train)

    def itterate_through_data(self, model):
        for Xtra,Xtes,f in self.dp.itterator:
            Xtra, Xtres = normalize_features(self.dp.Xtra, self.dp.Xtres)
            for name, model in self.models:
                self.train(model, Xtra, self.dp.train_y)
                train_pred = model.predict(Xtra)
                test_pred = model.predict(Xtes)
                # convert from kt [-] back to irradiance [W/m^2]
                convert_units(train_pred, self.dp.train_clear)
                convert_units(test_pred, self.dp.test_clear)
                # removes nighttime values (solar elevation < 5)
                remove_nighttime_values(train_pred, self.dp.train_clear)
                remove_nighttime_values(test_pred, self.dp.test_clear)
                self.dp.train.insert(self.dp.train.shape[1], "{}_{}_{}".format(self.dp.target, name,f), train_pred)
                self.dp.test.insert(self.dp.test.shape[1], "{}_{}_{}".format(self.dp.target, name,f), test_pred)
        