import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from typing import Union

ALL_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(
            self,
            feature_encoder: LabelEncoder,
            features: list[str] = ["Close"],
            seq_len: int = 10,
            pred_len: int = 1,
        ) -> None:
        """A transformer that transform the time series
        to the format suitable for LSTM model.
        """
        
        super().__init__()
        
        self.feature_encoder = feature_encoder
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def fit(
            self, 
            ts: np.ndarray,
            split_X_y: bool = True
        ):
        
        # make sure "Close" is at the end of features
        if "Close" in self.features:
            self.features.remove("Close")
        self.features.append("Close")
        
        # select features of interest
        ts = ts[:, self.feature_encoder.transform(self.features)]
        
        # fit min-max scaler to the data
        self.scaler.fit(ts)
        
        return self
    
    def transform(
            self, 
            ts: np.ndarray,
            split_X_y: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:
        
        # select features of interest
        ts = ts[:, self.feature_encoder.transform(self.features)]

        # scale the values of the time series
        scaled_ts = self.scaler.transform(ts)
        
        # convert to the format suitable for LSTM
        window_size = self.seq_len + self.pred_len if split_X_y else self.seq_len
        X_y = np.array([
            scaled_ts[i : i + window_size]
            for i in range(len(scaled_ts) - window_size + 1)
        ])
        
        if split_X_y:
            
            X = X_y[:, :-self.pred_len, :]
            y = X_y[:, -self.pred_len:, -1]
            
            return X, y
        
        else:
            
            X = X_y
            return X
        
class LSTM(nn.Module):
    
    def __init__(
            self,
            input_size: int, 
            hidden_size: int,
            num_layers: int, 
            output_size: int
        ) -> None:
        
        # initialize super class
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        # fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Notes:
        ------
        - `output` has shape (N, L, H)
        - `h_n` has shape (`num_layers`, N, H)
        
        where 
        - N is the batch size,
        - L is the sequence length, and
        - H is the hidden size
        """
        
        output, (h_n, c_n) = self.lstm.forward(x)

        # in fact, we want the last hidden value
        # from the last LSTM layer, i.e., h_n[-1, :, :]
        h = h_n[-1, :, :]

        # get predicted value from
        # the fully connected layer
        y = self.fc.forward(h)

        return y
    
def train(
        model: LSTM, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        num_epochs: int = 10,
        lr: float = 0.01,
        quiet=True
    ) -> float:
    
    # convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    
    # loss function
    criterion = nn.MSELoss(reduction="mean")
    
    # optimizer
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    for i in range(num_epochs):
        
        # predicted value
        y_pred = model.forward(X_train)

        # calculate loss
        loss: torch.Tensor = criterion(y_pred, y_train)
        
        if not quiet:
            print(f"epoch:\t{i}\tloss:\t{loss}")

        # clear gradients
        optimiser.zero_grad()

        # compute gradients through backward propagation
        loss.backward()

        # update parameters
        optimiser.step()
class PricePredictor(BaseEstimator):
    
    feature_encoder: LabelEncoder = None
    
    def __init__(
            self,
            features: list[str] = ["Close"],
            seq_len: int = 10,
            pred_len: int = 1,
            num_days_ago: int = 100,
            hidden_size: int = 32,
            num_layers: int = 1,
            num_epochs: int = 10,
            lr: float = 0.01   
        ) -> None:
        """A model that takes in M days to predict N-day closing prices 
        where M is `seq_len` and N is `pred_len`.

        Parameters
        ----------
            features (list[str], optional): Features of interest. Defaults to ["Close"].
            seq_len (int, optional): The number of days needed to predict prices. Defaults to 10.
            pred_len (int, optional): The number of days of closing prices to predict. Defaults to 1.
            num_days_ago (int, optional): The latest few days of the provided trainig date to consider. Defaults to 100.
            hidden_size (int, optional): Hidden size of LSTM. Defaults to 32.
            num_layers (int, optional): Number of layers of LSTM. Defaults to 1.
            num_epochs (int, optional): Number of epochs to train. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 0.01.

        Raises
        ------
            Exception: Missing feature encoder.
        """
        
        super().__init__()
        
        # initialize feature encoder
        if PricePredictor.feature_encoder is None:
            raise Exception("Feature encoder must be proveded as a class attribute.")
        
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_days_ago = num_days_ago
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.lr = lr
        
        # underlying LSTM model
        self.model: LSTM = None
        
        # time series transformer
        self.ts_transformer: TimeSeriesTransformer = None
        
        # the last sequence of input time series with length `self.seq_len`
        self.last_seq: np.ndarray = None
        
        # metric / method of scoring
        self.metric: str = "R2"
        
        # R2 score
        self.rs: float = None
        
        # mean squared error
        self.mse: float = None
        
    def _get_input_size(self) -> int:
        """Find the input size for the LSTM model.

        Returns
        -------
            int: Input size.
        """
        
        input_size = len(self.features)
        if "Close" not in self.features:
            input_size += 1
        return input_size
    
    def fit(self, ts: np.ndarray, quiet=True):
        """Fit the model.

        Parameters
        ----------
            ts (np.ndarray): Training time series.
            quiet (bool, optional): If `quiet` is `False` then training losses will be printed in the terminal. 
            Defaults to True.
        """
        
        # initialize LSTM model
        self.model = LSTM(
            input_size=self._get_input_size(),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.pred_len
        )
        
        # initialize time series transformer
        self.ts_transformer = TimeSeriesTransformer(
            feature_encoder=self.feature_encoder,
            features=self.features,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        )
        
        # only keep the last few days
        ts = ts[-self.num_days_ago:]
        
        # store the last sequence of the time series for prediction
        self.last_seq = ts[-self.seq_len:].copy()
        
        # get X and y
        X, y = self.ts_transformer.fit_transform(ts)
        
        # train LSTM model
        train(
            model=self.model,
            X_train=X,
            y_train=y,
            num_epochs=self.num_epochs,
            lr=self.lr,
            quiet=quiet
        )
    
    def score(self, test_ts: np.ndarray) -> float:
        """Compute the fitting score.

        Parameters
        ----------
            test_ts (np.ndarray): Test time seires.

        Returns
        -------
            float:
            
                1. R2 score is returned by default.
                2. If the computation of R2 score fails, then the negative MSE is returned.
        """
        
        # get true prices
        test_ts = test_ts[:self.pred_len]
        test_ts = test_ts[:, self.feature_encoder.transform(["Close"])]
        y_true = test_ts.flatten()

        # predicted prices
        y_pred = self.predict()
        
        # mean squred error
        self.mse = mean_squared_error(y_true, y_pred)
        
        try:
            self.r2 = r2_score(y_true, y_pred)
            return self.r2
        except:
            self.metric = "Negative MSE"
            return -self.mse
    
    def predict(
            self, 
            ts: Union[np.ndarray, None] = None
        ) -> np.ndarray:
        """Predict the prices of future N days if no input is provided
        where n equals the attribute `pred_len`. 
        If a time series (M days) is passed to this function, 
        then each time the function will take one of M days 
        to predict the prices of the next N days. 
        So, there will be (M + 1) batches of N-day prices.

        Parameters
        ----------
            ts (Union[np.ndarray, None], optional): A time series consisting of future data. 
            Defaults to None.

        Returns
        -------
            np.ndarray: 
            
                1. If the input is `None`, then a 1-D NumPy array of future N-day prices is returned.
                2. If there is an input time series, then an (M + 1)-by-N NumPy array is returned.
        """
        
        # a flag indicating whether there is an input time series
        is_empty_input = ts is None
        
        if is_empty_input:
            ts = self.last_seq
        else:
            # attach the input time series
            # to the last sequence of data
            ts = np.concatenate((self.last_seq, ts))
        
        # transform the time series
        X = self.ts_transformer.transform(ts, split_X_y=False)
        
        # convert to tensor
        X = torch.Tensor(X).detach()
              
        # predit with LSTM model
        y_pred: torch.Tensor = self.model.forward(X)

        # convert to NumPy array
        y_pred = y_pred.detach().numpy()
        
        # rescale the predicted values to prices
        sliding_prices: np.ndarray = np.apply_along_axis(
            self._to_price, 
            axis=1, 
            arr=y_pred
        )
        
        if is_empty_input:
            # return a 1-D array of future N-day prices
            price = sliding_prices[0]
            return price
        
        else:
            # return (M + 1) baches of N-day prices
            return sliding_prices
    
    def _to_price(self, y_pred: np.ndarray) -> np.ndarray:
        """Recover the price from the scaled data.

        Parameters
        ----------
            y_pred (np.ndarray): Predicted values ranging from -1 to 1.

        Returns
        -------
            np.ndarray: Prices.
        """
        
        price_min = self.ts_transformer.scaler.data_min_[-1]
        price_range = self.ts_transformer.scaler.data_range_[-1]
        feature_min, feature_max = self.ts_transformer.scaler.feature_range
        price = price_min + (y_pred - feature_min) * price_range / (feature_max - feature_min)
        return price