# Making Money in Stocks



## GitHub Repository and Webpage

Our source code is also available on the GitHub repository: `https://github.com/Isaac-Fate/stox`.

We have deployed a website that serves as both an outline and documentation for the project. The URL is `https://isaac-fate.github.io/stox`.



## Requirements

The Python version we use for our project is `Python 3.10.6`.

The requirements are described in file `requirements.txt`:

```
beautifulsoup4==4.11.1
click==8.1.3
joblib==1.1.0
numpy==1.22.4
pandas==1.4.4
requests==2.28.1
scikit_learn==1.1.3
torch==1.13.0
```


## Directory Structure

### Overview

```
.
├── __init__.py
├── main.py
├── lstm.py
├── strat.py
├── profit.py
├── news.py
├── book
│   ├── _config.yml
│   ├── _static
│   ├── _toc.yml
│   ├── closing-price-prediction
│   ├── data-collecting-and-visualization
│   ├── figures
│   ├── icon.jpg
│   ├── intro.md
│   ├── news-investigation
│   └── trading-strategy
├── data
│   ├── companies.json
│   ├── news.db
│   └── stocks.csv
├── models
│   ├── AAPL-3-day-predictor.pkl
│   ├── BA-3-day-predictor.pkl
│   ├── BAC-3-day-predictor.pkl
│   ├── META-3-day-predictor.pkl
│   ├── TSLA-3-day-predictor.pkl
│   ├── UNH-3-day-predictor.pkl
│   └── XOM-3-day-predictor.pkl
├── requirements.txt
└── README.md
```

### Our Modules

The central code files/modules are placed directly under the root directory.

- `lstm.py` mainly contains an LSTM model, a self-defined `PricePredictor` for prediction and functions to train and load models.
- `strat.py` contains two trading strategies: 1. Simple Moving Average (SMA); 2. Prediction-based strategy using pre-trained LSTM model. It also has a function to determine the best portfolio by assigning the weight of money to invest to each stock.
- `profit.py` contains a function to calculate the profit given the dates to buy and sell.
- `news.py` implements functions to automatically scrape news headlines, and then insert them into the SQLite database. And it contains fetch news headlines from the database.
- `main.py` is a command line app that implements commands mainly to train models, display project results (money made from the stocks) and scrape news headlines.

### Notebooks

Jupyter notebooks are placed under `./book/`. However, it is recommended to visit our website `https://isaac-fate.github.io/stox` to view these notebooks easily.

### Data

- `stocks.csv` A CSV file contains stock prices.
- `companies.json` A list of the most profitable companies from each sector.
- `news.db` A SQLite database storing the scraped news headlines.

### Pre-Trained Models

We have trained a `PricePredictor` (based on LSTM) for the most profitable company from each sector. The models are placed under `./models/`.



## How to Run?

The first thing to do is install all the requirements. 

```sh
pip install -r requirements.txt
```

We have implemented a command line app `main.py` to display part of the project results (more results can be viewed on our website). 

### Help Message

```
> python main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  find-company-news  Find news headlines for the specified company.
  report-profit      Report the profits using SMA strategy and...
  train              Train an LSTM model to predict future stock prices...
```

There are mainly 3 commands:
- `train` Train an LSTM model to predict future stock prices for a company.
- `report-profit` Report the profits using the SMA strategy and prediction-based strategy (LSTM), respectively.
- `find-company-news` Find news headlines for the specified company.

For each command, one can type 
```
> python main.py <COMMAND> --help
```
to see the help message for that command.

### Train Models

Use randomized search to train an LSTM model and save it.

```
python main.py train --company TSLA --pred-len 3
Training started...
Fitting 2 folds for each of 10 candidates, totalling 20 fits
[CV] END features=['Close'], hidden_size=16, num_days_ago=400, num_layers=1, seq_len=90; total time=   1.9s
[CV] END features=['Close'], hidden_size=16, num_days_ago=400, num_layers=1, seq_len=90; total time=   2.1s
[CV] END features=['Close'], hidden_size=32, num_days_ago=500, num_layers=1, seq_len=60; total time=   3.1s
[CV] END features=['Close'], hidden_size=32, num_days_ago=500, num_layers=1, seq_len=60; total time=   2.5s
[CV] END features=['Close'], hidden_size=16, num_days_ago=300, num_layers=2, seq_len=60; total time=   2.3s
[CV] END features=['Close'], hidden_size=16, num_days_ago=300, num_layers=2, seq_len=60; total time=   2.6s
[CV] END features=['Close'], hidden_size=16, num_days_ago=400, num_layers=1, seq_len=60; total time=   1.5s
[CV] END features=['Close'], hidden_size=16, num_days_ago=400, num_layers=1, seq_len=60; total time=   1.4s
[CV] END features=['Close'], hidden_size=32, num_days_ago=400, num_layers=2, seq_len=30; total time=   2.6s
[CV] END features=['Close'], hidden_size=32, num_days_ago=400, num_layers=2, seq_len=30; total time=   3.0s
[CV] END features=['Close', 'Open'], hidden_size=32, num_days_ago=300, num_layers=2, seq_len=90; total time=   5.8s
[CV] END features=['Close', 'Open'], hidden_size=32, num_days_ago=300, num_layers=2, seq_len=90; total time=   5.5s
[CV] END features=['Close', 'Open', 'Volume'], hidden_size=16, num_days_ago=300, num_layers=2, seq_len=30; total time=   1.3s
[CV] END features=['Close', 'Open', 'Volume'], hidden_size=16, num_days_ago=300, num_layers=2, seq_len=30; total time=   1.3s
[CV] END features=['Close'], hidden_size=16, num_days_ago=500, num_layers=1, seq_len=60; total time=   1.8s
[CV] END features=['Close'], hidden_size=16, num_days_ago=500, num_layers=1, seq_len=60; total time=   1.6s
[CV] END features=['Close'], hidden_size=32, num_days_ago=500, num_layers=2, seq_len=90; total time=   7.8s
[CV] END features=['Close'], hidden_size=32, num_days_ago=500, num_layers=2, seq_len=90; total time=   8.9s
[CV] END features=['Close', 'Open'], hidden_size=32, num_days_ago=400, num_layers=2, seq_len=30; total time=   3.0s
[CV] END features=['Close', 'Open'], hidden_size=32, num_days_ago=400, num_layers=2, seq_len=30; total time=   3.0s
Training complete!
Model is saved as ./models/TSLA-3-day-predictor.pkl.
```

### Report Profit

This display the major result of our project. We compare how much money we can gain by applying the two different strategies: Simple Moving Average (SMA) and prediction-based (LSTM) strategy. Of course, we prefer the latter.

```
> python main.py report-profit --capital 10000
Report:
----------------------------------------------------------------
                 META    TSLA    XOM    BAC    UNH     BA   AAPL
SMA           -19.97%   9.19% -2.80% -2.12%  0.77%  0.00% 17.77%
LSTM            9.44% -13.08%  2.06% 11.74%  0.19%  8.91% 15.18%
Weight         14.70%   0.87% 25.35%  3.45% 26.79% 28.44%  0.39%
Weighted SMA   -2.94%   0.08% -0.71% -0.07%  0.21%  0.00%  0.07%
Weighted LSTM   1.39%  -0.11%  0.52%  0.41%  0.05%  2.54%  0.06%
----------------------------------------------------------------
Total profit using SMA: -336.22
Total profit using LSTM: 484.81
================================================================
```

### Find News Headlines

Search and insert the news headlines into the SQLite database.

```
> python main.py find-company-news --company TSLA --query "Tesla" --from-date "2022-10-1" --to-date "2022-10-5" --force 
Start searching for news headlines...
News on 2022-10-01: Tesla boss Elon Musk presents humanoid robot Optimus
News on 2022-10-02: Tesla blames logistics problems after delivering fewer cars than forecast
News on 2022-10-03: Tesla slides on widening delivery and production gap, demand worries
News on 2022-10-04: A Musk Retweet: Tesla CEO Says He'll Pay $44 Billion to Buy Twitter
News on 2022-10-05: Musk's move to close Twitter deal leaves Tesla investors worried
Successfully found all the news for TSLA.
```

