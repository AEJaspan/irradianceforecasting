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

class ThetaLogistic(ssm.StateSpaceModel):
    """ Theta-Logistic state-space model (used in Ecology).
    """
    default_params = {'tau0':.15, 'tau1':.12, 'tau2':.1, 'sigmaX': 0.47, 'sigmaY': 0.39}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def f(self, x):
        return (x + self.tau0 - self.tau1 * np.exp(self.tau2 * x))

    def PX(self, t, xp):  #  Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=self.f(xp), scale=self.sigmaX)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Normal(loc=x, scale=self.sigmaY)

class ThetaLogistic_with_prop(ThetaLogistic):
    def proposal0(self, data):
        return self.PX0()
    def proposal(self, t, xp, data):
        prec_prior = 1. / self.sigmaX**2
        prec_lik = 1. / self.sigmaY**2
        var = 1. / (prec_prior + prec_lik)
        mu = var * (prec_prior * self.f(xp) + prec_lik * data[t])
        return dists.Normal(loc=mu, scale=np.sqrt(var))


class ThetaLogistic_with_upper_bound(ThetaLogistic_with_prop):
    def upper_bound_log_pt(self, t):
        return -np.log(np.sqrt(2 * np.pi) * self.sigmaX)




# Importing the libraries
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

class Dataset(torch.utils.data.Dataset):
    def __init__(self, src, trg_in, trg_out):
        self.src = src
        self.trg_in = trg_in
        self.trg_out = trg_out

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return self.src, self.trg_in, self.trg_out
        
def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask

class TimeSeriesForcasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        channels=512,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src, trg = x

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

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

    def forecast_ols(self, model, Xtra, ytra):
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
        # # Define and create a custom state-space model instance
        # mu = -1.0
        # rho = 0.9
        # sigma = 0.1
        # self.prior_dict = {'mu': mu,
        #                    'rho' : rho,
        #                    'sigma' : sigma}
        # custom_model = CustomStateSpaceModel(**self.prior_dict)
        # # Create a Bootstrap filter using the custom model and normalized data
        # fk_model = ssm.Bootstrap(ssm=custom_model, data=ytra)
        # self.pf = particles.SMC(fk=fk_model, N=1, resampling='stratified',
        #                         collect=[Moments()], store_history=True)

        # # Run the particle filter
        # self.pf.run()
        # return self.pf
        
        
        
        # my_better_ssm = ThetaLogistic_with_prop()
        # fk_guided = ssm.GuidedPF(ssm=my_better_ssm, data=y)
        # self.pf = particles.SMC(fk=fk_guided, N=100, collect=[Moments()])
        # self.pf.run()
        # return self.pf
        # my_ssm = ThetaLogistic_with_upper_bound()
        # alg = particles.SMC(fk=ssm.GuidedPF(ssm=my_ssm, data=ytra),
        #                     N=100, store_history=True)
        # alg.run()
        # (more_trajectories, acc_rate) = alg.hist.backward_sampling(10, linear_cost=True,
        #                                                    return_ar=True)
        # return alg
        # Defining the model
        # Defining the model
        model_better = keras.Sequential([
            keras.layers.Dense(16, input_shape=(Xtra.shape[1],), activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        x, y = Xtra, ytra
        # Compiling the model
        model_better.compile(optimizer='adam',
                            loss='mean_absolute_error')        
        # fitting the model

        model_better.fit(x, y, epochs=10, batch_size=8)
        return model_better

    def forecast_pf(self, pf, Xtra, ytra):
        # # Access estimated states
        # estimated_states = pf.hist.X 
        # # # remove burnin
        # # burnin=100
        # # for i, param in enumerate(self.prior_dict.keys()):
        # #     pf.chain.theta[param][burnin:]
        # # Define forecasting horizon
        # forecast_horizon = Xtra.shape[0] # 10  # Adjust as needed

        # # Initialize arrays to store forecasts
        # forecasts = []
        # forecast_std_devs = []
        # # Generate forecasts for each time step in the forecasting horizon
        # for t in range(forecast_horizon):
        #     # Use the estimated volatility to generate a forecast for the next observation
        #     # Assuming 'estimated_states' contains the estimated volatilities
        #     estimated_volatility = estimated_states[t]  # Replace with the appropriate index if needed
        #     # Generate a forecast for the next observation based on the estimated volatility
        #     # For example, you can use a normal distribution with mean 0 and standard deviation 'estimated_volatility'
        #     forecast = np.random.normal(loc=0, scale=np.abs(estimated_volatility))
            
        #     # Append forecast to the forecasts array
        #     forecasts.append(forecast)
            
        #     # Append the estimated volatility to the forecast_std_devs array for visualization
        #     forecast_std_devs.append(estimated_volatility)
        # return np.array(forecasts).reshape(-1) # , np.array(forecast_std_devs).reshape(-1)

        # return [m['mean'] for m in pf.summaries.moments]
        return pf.predict(Xtra).reshape(-1)
    
    
    
    def train_transformer(self, Xtra, ytra):
        # channels = Xtra.shape[1]
        # dropout=0.1 #default
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=channels,
        #     nhead=8,
        #     dropout=dropout,
        #     dim_feedforward=4 * channels,
        # )
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=channels,
        #     nhead=8,
        #     dropout=dropout,
        #     dim_feedforward=4 * channels,
        # )

        # self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)
    
        # self.input_projection = Linear(channels, channels)
        # self.output_projection = Linear(channels, channels)

        # self.linear = Linear(channels, 1)

        # self.do = nn.Dropout(p=dropout)
        model = TimeSeriesForcasting(
            n_encoder_inputs=Xtra.shape[1] + 1,
            n_decoder_inputs=Xtra.shape[1] + 1,
            lr=1e-5,
            dropout=0.1,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            dirpath='.',
            filename="ts",
        )
        max_epochs=1
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            log_every_n_steps=100,
            callbacks=[checkpoint_callback],
        )
        y_lag = np.concatenate([[np.nan], ytra[:-1]]).reshape(-1,1)
        ytra = ytra.reshape(-1,1)
        trg_in = np.hstack([Xtra, y_lag])
        src = np.hstack([Xtra, ytra])
        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(ytra, dtype=torch.float)
        train_loader = DataLoader(
            Dataset(src, trg_in, trg_out),
            batch_size=100,
            num_workers=4,
            shuffle=True,
        )
        trainer.fit(model, train_dataloaders=train_loader)
        input_weight = model.input_pos_embedding.weight.detach().numpy()
        target_weight = model.input_pos_embedding.weight.detach().numpy()
        nan_input_weight = np.count_nonzero(~np.isnan(input_weight))
        nan_target_weight = np.count_nonzero(~np.isnan(target_weight))
        print(f'n_input_weights = {input_weight.shape[0]*input_weight.shape[1]}')
        print(f'n_nan_input_weights = {nan_input_weight}')
        print(f'perc = {(nan_input_weight/input_weight.shape[0]*input_weight.shape[1])*100}')
        print(f'n_target_weights = {target_weight.shape[0]*target_weight.shape[1]}')
        print(f'n_nan_target_weights = {nan_target_weight}')
        print(f'perc = {(nan_target_weight/target_weight.shape[0]*target_weight.shape[1])*100}')
        return model

    def forecast_transformer(self, model, Xtra, ytra):
        # Xtra = torch.tensor(Xtra, dtype=torch.float)
        # ytra = torch.tensor(ytra, dtype=torch.float)
        y_lag = np.concatenate([[np.nan], ytra[:-1]]).reshape(-1,1)
        ytra = ytra.reshape(-1,1)
        trg_in = np.hstack([Xtra, y_lag])
        src = np.hstack([Xtra, ytra])
        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(ytra, dtype=torch.float)
        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)
        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
        # for j in range(1, horizon_size):
        #     last_prediction = prediction[0, -1]
        #     trg_in[:, j, -1] = last_prediction
        #     prediction = model((src, trg_in[:, : (j + 1), :]))

        return prediction

        
if __name__=='__main__':
    # import pandas as pd

    # from src import DataPipeline
    # from src import Model

    # dp = DataPipeline(dev_limit=5000)
    # dp.load_data()
    # targets  = ["dni", "ghi"]
    # horizons = ["5min", "10min", "15min", "20min", "25min", "30min"]
    # results=pd.DataFrame()
    # for t in targets[:1]:
    #     h = horizons[-1]
    #     target, horizon = t, h
    #     dp.train_test_split(target, horizon)
    #     model = Model(dp)
    #     summary = model.itterate_through_data()
    #     results = pd.concat([results,summary])
    # print(results)
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from src import DataPipeline
    from src import Model
    from src import IrradianceForecastingModel
    from src import plot_skill
    from src import (
        normalize_features,
        convert_units,
        remove_nighttime_values
    )
    targets  = ["dni", "ghi"]
    horizons = ["5min", "10min", "15min", "20min", "25min", "30min"]

    t = targets[0]
    h = horizons[-1]
    target, horizon = t, h
    dp = DataPipeline(dev_limit=5)
    dp.load_data()
    train_test_split = dp.train_test_split(target, horizon)
    model = IrradianceForecastingModel()
    models = [
                ["trf", model.train_transformer, model.forecast_transformer],
                ["PF", model.train_particle_filter, model.forecast_pf],
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
    # Xtra, Xtes = normalize_features(Xtra, Xtes)
    m = train_fn(Xtra, dp.train_y)
    train_pred = forecast_fn(m, Xtra, dp.train_y)
    print(train_pred)