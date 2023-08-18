from sklearn.preprocessing import StandardScaler
import numpy as np

def normalize_features(Xtra, Xtes):
    """
    `Xtra` is a variable that represents the training features used for forecasting. It is a
    subset of the input features (`train_X` or `train_X_endo`) that are used to train the
    forecast models. The features are normalized using `StandardScaler` before training the
    models.
    
    `Xtes` is a variable that represents the test set features. It is
    used to store the feature values for the test set, which will be
    used to make predictions using the trained models.
    """
    scaler = StandardScaler()
    scaler.fit(Xtra)
    Xtra = scaler.transform(Xtra)
    Xtes = scaler.transform(Xtes)
    
    return Xtra, Xtes

def convert_units(predictions, clear_sky_index):
    """
    convert from kt [-] back to irradiance [W/m^2]
    where k_{t} is the clear sky index
    """
    predictions *= clear_sky_index


def remove_nighttime_values(predictions, elevation):
    """
    removes nighttime values (solar elevation < 5)
    """
    predictions[elevation < 5] = np.nan
