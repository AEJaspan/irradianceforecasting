import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax
from pykalman import UnscentedKalmanFilter

class IrradianceForecastingModel:
    def __init__(self):
        self.sarimax_model = None
        self.pf_model = None

    def train_sarimax(self, Xtra, ytra, order, seasonal_order, exog=None):
        self.sarimax_model = sarimax.SARIMAX(ytra, exog=exog, order=order, seasonal_order=seasonal_order)
        self.sarimax_results = self.sarimax_model.fit()

    def train_particle_filter(self, Xtra, ytra, initial_state_mean, initial_state_covariance, transition_covariance, observation_covariance):
        self.pf_model = UnscentedKalmanFilter(initial_state_mean=initial_state_mean,
                                              initial_state_covariance=initial_state_covariance,
                                              transition_covariance=transition_covariance,
                                              observation_covariance=observation_covariance)
        self.pf_model = self.pf_model.em(ytra).smooth(ytra)

    def evaluate(self, Xtes, ytes):
        if self.sarimax_model is not None:
            sarimax_forecasts = self.sarimax_results.forecast(steps=len(ytes), exog=Xtes)
            sarimax_metrics = summary_stats(ytes, sarimax_forecasts)
            print("SARIMAX Metrics:")
            print(sarimax_metrics)

        if self.pf_model is not None:
            pf_means, pf_covs = self.pf_model.filter(ytes)
            pf_forecasts = pf_means[:, 0]  # Assuming irradiance is the first dimension
            pf_metrics = summary_stats(ytes, pf_forecasts)
            print("Particle Filter Metrics:")
            print(pf_metrics)

# Train SARIMAX model
sarimax_order = (1, 1, 1)
sarimax_seasonal_order = (1, 1, 1, 24)  # Example seasonal order for daily data
model.train_sarimax(data_pipeline.Xtra, ytra, sarimax_order, sarimax_seasonal_order, exog=data_pipeline.Xtra)

# Train Particle Filter model
initial_state_mean = np.zeros(1)  # Assuming irradiance is the first dimension
initial_state_covariance = np.eye(1)
transition_covariance = np.eye(1)
observation_covariance = np.eye(1)
model.train_particle_filter(data_pipeline.Xtra, ytra, initial_state_mean, initial_state_covariance, transition_covariance, observation_covariance)
