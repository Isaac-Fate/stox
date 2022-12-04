# %% [markdown]
# # Scraping Stock Data

# %% [markdown]
# This notebook download the stock data from Yahoo.

# %%
import pandas as pd
import io
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import os

# %%
# get stock information to determine which sector it belongs to
aapl = yf.Ticker("AAPL")
aapl.info

# %%
# Technology 10
technology_tickers = ["AAPL","MSFT","NVDA","TSM","ORCL","ASML","AVGO","CSCO","ACN","IBM"]
technology_dict = {}
for t in technology_tickers:
    technology_dict[t] = yf.download(tickers=t, period='5y')
    technology_dict[t].drop(columns=["Adj Close"], inplace = True)
    technology_dict[t].insert(0, "Company",t)
    technology_dict[t].insert(1, "Sector", "Technology")

# %%
technology_dict["AAPL"]

# %%
# Healthcare 10
healthcare_tickers = ["UNH","JNJ","LLY","PFE","ABBV","MRK","NVO","TMO","DHR","AZN"]
healthcare_dict = {}
for t in healthcare_tickers:
    healthcare_dict[t] = yf.download(tickers=t, period='5y')
    healthcare_dict[t].drop(columns=["Adj Close"], inplace = True)
    healthcare_dict[t].insert(0, "Company",t)
    healthcare_dict[t].insert(1, "Sector", "Healthcare")

# %%
healthcare_dict["UNH"]

# %%
# Consumer Cyclical 12
cyclical_tickers = ["AMZN","TSLA","HD","MCD","TM","BABA","NKE","LOW","SBUX","ABNB","LULU","EBAY"]
cyclical_dict = {}
for t in cyclical_tickers:
    cyclical_dict[t] = yf.download(tickers=t, period='5y')
    cyclical_dict[t].drop(columns=["Adj Close"], inplace = True)
    cyclical_dict[t].insert(0, "Company",t)
    cyclical_dict[t].insert(1, "Sector", "Consumer Cyclical")

# %%
cyclical_dict["AMZN"]

# %%
# Industrials 2
industrials_tickers = ["RTX","BA"]
industrials_dict = {}
for t in industrials_tickers:
    industrials_dict[t] = yf.download(tickers=t, period='5y')
    industrials_dict[t].drop(columns=["Adj Close"], inplace = True)
    industrials_dict[t].insert(0, "Company",t)
    industrials_dict[t].insert(1, "Sector", "Industrials")

# %%
industrials_dict["RTX"]

# %%
# Financial Services 8
financial_tickers = ["JPM","BAC","GS","MS","V","HSBC","C","UBS"]
financial_dict = {}
for t in financial_tickers:
    financial_dict[t] = yf.download(tickers=t, period='5y')
    financial_dict[t].drop(columns=["Adj Close"], inplace = True)
    financial_dict[t].insert(0, "Company",t)
    financial_dict[t].insert(1, "Sector", "Financial Services")

# %%
financial_dict["C"]

# %%
# Communication Services 5
communication_tickers = ["GOOG","META","NFLX","BIDU","DIS"]
communication_dict = {}
for t in communication_tickers:
    communication_dict[t] = yf.download(tickers=t, period='5y')
    communication_dict[t].drop(columns=["Adj Close"], inplace = True)
    communication_dict[t].insert(0, "Company",t)
    communication_dict[t].insert(1, "Sector", "Communication Services")

# %%
communication_dict["GOOG"]

# %%
# Energy 4
energy_tickers = ["XOM","CVX","SHEL","COP"]
energy_dict = {}
for t in energy_tickers:
    energy_dict[t] = yf.download(tickers=t, period='5y')
    energy_dict[t].drop(columns=["Adj Close"], inplace = True)
    energy_dict[t].insert(0, "Company",t)
    energy_dict[t].insert(1, "Sector", "Energy")

# %%
energy_dict["SHEL"]

# %%
data = pd.concat(list(technology_dict.values())
          +list(healthcare_dict.values())
          +list(cyclical_dict.values())
          +list(industrials_dict.values())
          +list(financial_dict.values())
          +list(communication_dict.values())
          +list(energy_dict.values()))

# %%
data

# %%
data.isnull().sum()

# %% [markdown]
# Don't save it if data already exists:

# %%
filepath = "../../data/stocks.csv"
if not os.path.exists(filepath):
    data.to_csv("../../data/stocks.csv")


