import pandas as pd
from sklearn.preprocessing import LabelEncoder

import lstm

def sma(
        stock: pd.DataFrame,
        long_period: int = 15,
        short_period: int = 5
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Simple moving average strategy.

    Parameters
    ----------
        stock (pd.DataFrame): A data frame containing stock prices. 
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
    long = stock.rolling(long_period).mean()
    short = stock.rolling(short_period).mean()  
    
    # dates of golden and death crosses
    golden = ((short > long) & ((short < long).shift(1))).query("Close == True").index
    death = ((short < long) & ((short > long).shift(1))).query("Close == True").index
    
    return golden, death

def trade_by_pred(
        stock: pd.DataFrame,
        price_predictor: lstm.PricePredictor,
        num_days_ahead: int = 1
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    
    # prerequisites
    if num_days_ahead > price_predictor.pred_len:
        raise Exception("The provided price predictor is unable to predict that far.")
    
    # set up feature encoder for the price predictor
    feature_encoder = LabelEncoder()
    feature_encoder.fit(lstm.ALL_FEATURES)
    lstm.PricePredictor.feature_encoder = feature_encoder
    
    # select the features of interest in data frame and transform it to NumPy array
    stock = stock[lstm.ALL_FEATURES]
    ts = stock[feature_encoder.classes_].to_numpy()
    
    # use price predictor to predict future prices
    sliding_prices = price_predictor.predict(ts)
    future_prices = sliding_prices[1:, num_days_ahead-1]
    
    # set up helper columns
    stock = stock.copy()
    stock["Future"] = future_prices
    
    # Find buying and selling dates
    buy_dates = []
    sell_dates = []
    is_holding_shares = False
    for date, row in stock.iterrows():
        if is_holding_shares:
            # price will go down, sell it!
            if row["Future"] < row["Close"]:
                sell_dates.append(date)
                is_holding_shares = False
        else:
            # price will go up, buy!
            if row["Future"] > row["Close"]:
                buy_dates.append(date)
                is_holding_shares = True

    # convert to Pandas `DatetimeIndex`
    buy_dates = pd.DatetimeIndex(buy_dates)
    sell_dates = pd.DatetimeIndex(sell_dates)

    return buy_dates, sell_dates