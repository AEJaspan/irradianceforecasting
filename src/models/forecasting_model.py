# Importing the libraries
from tensorflow import keras
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from tensorflow.keras.saving import load_model as load_keras_model
from joblib import dump, load
from root import ROOT_DIR, Path


def nn_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(16, input_shape=input_shape, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    # Compiling the model
    model.compile(optimizer='adam',
                  loss='mean_absolute_error')
    return model


class IrradianceForecastingModel:
    """
    A class for training and evaluating forecasting models
    for irradiance data using ordinary least squares
    regression (OLS) and a Particle Filter.
    """
    def __init__(
                    self, target, horizon, 
                    from_pretrained=False, save_models=False
                ) -> None:
        self.from_pretrained = from_pretrained
        self.save_models = save_models
        self.target = target
        self.horizon = horizon
        self.model_dir = ROOT_DIR/f'models/{target}/{horizon}'
        if save_models:
            self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_ols(self, Xtra, ytra, f):
        """
        Train the OLS forecasting model.

        Args:
            Xtra (array-like): Exogenous features for training.
            ytra (array-like): Target variable for training.

        Returns:
            None
        """
        model_name = f'{f}_ols_model'
        if self.from_pretrained:
            model = self.load_model(model_name)
        else:
            model = linear_model.LinearRegression()
            model.fit(Xtra, ytra)
            self.save_model(model_name, model)
        return model

    def forecast_ols(self, model, Xtra):
        return model.predict(Xtra)

    def train_nn(self, Xtra, ytra, f):
        """
        Train the Neural Network model.

        Args:
            Xtra (array-like): Exogenous features for training.
            ytra (array-like): Target variable for training.
        Returns:
            None
        """
        model_name = f'{f}_nn_model'
        if self.from_pretrained:
            model = self.load_model(model_name)
        else:
            model = nn_model((Xtra.shape[1],))
            model.fit(Xtra, ytra, epochs=10, batch_size=8)
            self.save_model(model_name, model)
        return model

    def forecast_nn(self, nn, Xtra):
        return nn.predict(Xtra).reshape(-1)

    def save_model(self, name, model):
        if not self.save_models:
            return None
        filepath = self.model_dir / name
        if type(model) == Sequential:
            filepath = filepath.with_suffix('.h5')
            model.save(filepath)
        else:
            filepath = filepath.with_suffix('.joblib')
            with filepath.open('wb') as f:
                dump(model, f)
        return None

    def load_model(self, name):
        filepath = self.model_dir / name
        if '_nn_model' in name:
            filepath = filepath.with_suffix('.h5')
            model = load_keras_model(filepath)
        else:
            filepath = filepath.with_suffix('.joblib')
            with filepath.open('rb') as f:
                model = load(f)
        return model


if __name__=='__main__':
    from src import DataPipeline
    from src import IrradianceForecastingModel
    from src import normalize_features


    targets  = ["dni", "ghi"]
    horizons = ["5min", "10min", "15min", "20min", "25min", "30min"]

    t = targets[0]
    h = horizons[-1]
    target, horizon = t, h
    dp = DataPipeline(dev_limit=5)
    dp.load_data()
    train_test_split = dp.train_test_split(target, horizon)
    model = IrradianceForecastingModel(from_pretrained=False)
    models = [
                ["nn", model.train_nn, model.forecast_nn],
                ["OLS", model.train_ols, model.forecast_ols],
            ]

    name, train_fn, forecast_fn = models[0]
    Xtra,Xtes,f = list(train_test_split.itterator)[1]


    cols = [
            "{}_{}".format(target,horizon),  # actual
            "{}_kt_{}".format(target,horizon),  # clear-sky index
            "{}_clear_{}".format(target,horizon),  # clear-sky model
            "elevation_{}".format(horizon)   # solar elevation 
        ]
    Xtra, Xtes = normalize_features(Xtra, Xtes)
    m = train_fn(Xtra, dp.train_y)
    train_pred = forecast_fn(m, Xtra)
    print(train_pred)