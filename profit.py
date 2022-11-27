import numpy as np
import pandas as pd
from typing import Union

def calc_profit(
        stock_price: pd.DataFrame,
        buy_dates: pd.DatetimeIndex,
        sell_dates: pd.DatetimeIndex,
        start_date: str,
        capital: Union[float, None] = None,
    ) -> float:
    """Calculate the total profit.

    Parameters
    ----------
        stock_price (pd.DataFrame): A data frame whose first column consists of closing prices.
        buy_dates (pd.DatetimeIndex): Dates to buy.
        sell_dates (pd.DatetimeIndex): Dates to sell.
        start_date (str): Starting date represented as a string. For examples, `"2022"`, `"2022-01"` or `"2022-01-01"`.
        capital (Union[float, None], optional): Capital principal, i.e., invested money. Defaults to None.

    Returns
    -------
        float: Profit or its rate if the capital is not provided.
    """
    
    # column name that stores the closing prices
    price_col = stock_price.columns[0]
    price_col = "Close"
    
    # only consider the dates after the provided starting date
    buy_dates = buy_dates[buy_dates >= start_date]
    sell_dates = sell_dates[sell_dates >= start_date]
    
    # gain nothing
    if len(buy_dates) == 0:
        return 0
    
    # if we need to sell first, then ignore the first selling date 
    # since we don't have any shares 
    if buy_dates[0] > sell_dates[0]:
        sell_dates = sell_dates[1:]
    
    buy = stock_price.loc[buy_dates, price_col].to_numpy()
    sell = stock_price.loc[sell_dates, price_col].to_numpy()
    
    # buying and selling dates should come in pairs 
    # except that there may be one more buying date in the end
    assert len(buy) == len(sell) or len(buy) == len(sell) + 1
    
    # don't buy
    if len(buy) == len(sell) + 1:
        buy = buy[:-1]
        
    # an array of percentage profit
    pct = (sell - buy) / buy
    
    # total amount
    capital = 1 if capital is None else capital
    total = capital * np.prod(1 + pct)
    
    # profit/gain
    profit = total - capital
    
    return profit