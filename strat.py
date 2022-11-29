import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import partial
from typing import Union

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
    """A trading strategy based on predicted stock prices.

    Parameters
    ----------
        stock (pd.DataFrame): A data frame containing stock prices. 
            It must contains a `"Close"` column.
        price_predictor (lstm.PricePredictor): A pretrained model.
        num_days_ahead (int, optional): Refer to the stock price in the N-th day ahead from now. Defaults to 1.

    Raises
    ------
        Exception: Missing feature encoder.
        
    Returns
    -------
        tuple[pd.DatetimeIndex, pd.DatetimeIndex]: 
        
            1. Dates to buy.
            2. Dates to sell.
    """
    
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

def determine_portfolio(
       stocks: pd.DataFrame,
       companies: list[str],
       num_days_left_out: int = 0,
       num_samples: int = 1000,
       random_seed: Union[int, None] = None
    ) -> np.ndarray:
    """Determine the most effective portfolio that maximize the Sharpe ratio.

    Parameters
    ----------
        stocks (pd.DataFrame): A complete data frame consists of stock information of all companies.
        companies (list[str]): A list of candidate companies.
        num_days_left_out (int, optional): Number of days left out from the end of the data frame. Defaults to 0.
        num_samples (int, optional): Number of samples to generate in the Monte Carlo simulation. Defaults to 1000.
        random_seed (Union[int, None], optional): Random seed used in Monte Carlo simulation. Defaults to None.

    Returns
    -------
        np.ndarray: Weight to be assigned to each company.
    """
    
    # get daily stock returns
    stock_returns = get_stock_returns(stocks, companies, num_days_left_out)
    
    # Monte Carlo simulation

    # set seed
    rng = np.random.RandomState(random_seed)

    # the weight for each company is selected with equal probabilities
    num_companies = len(companies)
    simulated_weights = rng.random((num_samples, num_companies))

    # calculate volatility for each portfolio
    volatilities = np.apply_along_axis(
        partial(calc_volatility, stock_returns),
        axis=1,
        arr=simulated_weights
    )

    # calculate stock return for each portfolio
    weighted_stock_returns = np.apply_along_axis(
        partial(calc_weighted_stock_return, stock_returns),
        axis=1,
        arr=simulated_weights
    )
    
    # calculate Sharpe ratio
    sharpe_ratio = weighted_stock_returns / volatilities
    
    # find the portfolio that maximize the sharpe ratio
    optimal_weight_ix = np.argmax(sharpe_ratio)
    optimal_weight: np.ndarray = simulated_weights[optimal_weight_ix]
    
    # scale the weights
    optimal_weight = optimal_weight / optimal_weight.sum()
    
    return optimal_weight
    

def get_stock_returns(
       stocks: pd.DataFrame,
       companies: list[str],
       num_days_left_out: int = 0 
    ) -> pd.DataFrame:
    
    df_list = []
    for company in companies:
        df = stocks.query(f"Company == '{company}'")[["Close"]]
        df.rename(columns={"Close": company}, inplace=True)
        df_list.append(df)

    # join stock prices of different companies
    df = pd.DataFrame.join(df_list[0], df_list[1:])

    # leave last 90 days out
    df = df[:-num_days_left_out]
    
    return df

def calc_volatility(stock_returns: pd.DataFrame, weight: np.ndarray) -> float:
    
    # scale weights
    weight = weight / weight.sum()
    
    # convert to column vector
    weight_vec = weight.reshape((-1, 1))
    
    # volatility
    volatility = np.sqrt(weight_vec.T @ np.cov(stock_returns.T) @ weight_vec)
    volatility = volatility.item()
    
    return volatility

def calc_weighted_stock_return(stock_returns: pd.DataFrame, weight: np.ndarray) -> float:
    
    # scale weights
    weight = weight / weight.sum()
    
    return np.dot(weight, stock_returns.mean().to_numpy())