import pandas as pd
from sklearn.preprocessing import LabelEncoder

import lstm

def sma(
        stock_price: pd.DataFrame,
        long_period: int = 15,
        short_period: int = 5
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Simple moving average strategy.

    Parameters
    ----------
        stock_price (pd.DataFrame): A data frame containing stock prices. 
            It must contains a `"Close"` column.
        long_period (int, optional): _description_. Defaults to 15.
        short_period (int, optional): _description_. Defaults to 5.

    Returns
    -------
        tuple[pd.DatetimeIndex, pd.DatetimeIndex]: 
        
            1. Golden crosses (dates to buy).
            2. Death crosses (dates to sell).
    """
    
    # calculate long and short-term SMA
    long = stock_price.rolling(long_period).mean()
    short = stock_price.rolling(short_period).mean()  
    
    # dates of golden and death crosses
    golden = ((short > long) & ((short < long).shift(1))).query("Close == True").index
    death = ((short < long) & ((short > long).shift(1))).query("Close == True").index
    
    return golden, death

def trade_by_pred(
        stock_price: pd.DataFrame,
        price_predictor: lstm.PricePredictor
    ):
    
    feature_encoder = LabelEncoder()
    feature_encoder.fit(lstm.ALL_FEATURES)
    lstm.PricePredictor.feature_encoder = feature_encoder
    
    features = stock_price.columns.intersection(lstm.ALL_FEATURES)
    features = features.intersection(price_predictor.features)
    ts = stock_price[features]
    
    y_pred = price_predictor.predict(ts)
    
    ...
    