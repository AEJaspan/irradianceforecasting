import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import Moments
from sklearn import linear_model


class CustomStateSpaceModel(ssm.StateSpaceModel):
    def PX0(self):
        # Define the distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho**2))
    
    def PX(self, t, xp):
        # Define the distribution of X_t given X_{t-1} = xp
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    
    def PY(self, t, xp, x):
        # Define the distribution of Y_t given X_t = x
        return dists.Normal(loc=0., scale=np.exp(x))


class IrradianceForecastingModel:
    """
    A class for training and evaluating forecasting models
    for irradiance data using ordinary least squares
    regression (OLS) and a Particle Filter.
    """

    def train_ols(self, Xtra, ytra):
        """
        Train the OLS forecasting model.

        Args:
            Xtra (array-like): Exogenous features for training.
            ytra (array-like): Target variable for training.

        Returns:
            None
        """
        model = linear_model.LinearRegression()
        model.fit(Xtra, ytra)

        return model

    def forecast_ols(self, model, Xtra):
        return model.predict(Xtra)

    def train_particle_filter(self, Xtra, ytra):
        """
        Train the Particle Filter forecasting model.

        Args:
            Xtra (array-like): Exogenous features for training.
            ytra (array-like): Target variable for training.
        Returns:
            None
        """
        # Define and create a custom state-space model instance
        mu = -1.0
        rho = 0.9
        sigma = 0.1
        self.prior_dict = {'mu': mu,
                           'rho' : rho,
                           'sigma' : sigma}
        custom_model = CustomStateSpaceModel(**self.prior_dict)
        # Create a Bootstrap filter using the custom model and normalized data
        fk_model = ssm.Bootstrap(ssm=custom_model, data=ytra)
        self.pf = particles.SMC(fk=fk_model, N=1, resampling='stratified',
                                collect=[Moments()], store_history=True)

        # Run the particle filter
        self.pf.run()
        return self.pf

    def forecast_pf(self, pf, Xtra):
        # Access estimated states
        estimated_states = pf.hist.X 
        # # remove burnin
        # burnin=100
        # for i, param in enumerate(self.prior_dict.keys()):
        #     pf.chain.theta[param][burnin:]
        # Define forecasting horizon
        forecast_horizon = Xtra.shape[0] # 10  # Adjust as needed

        # Initialize arrays to store forecasts
        forecasts = []
        forecast_std_devs = []
        # Generate forecasts for each time step in the forecasting horizon
        for t in range(forecast_horizon):
            # Use the estimated volatility to generate a forecast for the next observation
            # Assuming 'estimated_states' contains the estimated volatilities
            estimated_volatility = estimated_states[t]  # Replace with the appropriate index if needed
            # Generate a forecast for the next observation based on the estimated volatility
            # For example, you can use a normal distribution with mean 0 and standard deviation 'estimated_volatility'
            forecast = np.random.normal(loc=0, scale=np.abs(estimated_volatility))
            
            # Append forecast to the forecasts array
            forecasts.append(forecast)
            
            # Append the estimated volatility to the forecast_std_devs array for visualization
            forecast_std_devs.append(estimated_volatility)
        return np.array(forecasts).reshape(-1) # , np.array(forecast_std_devs).reshape(-1)
