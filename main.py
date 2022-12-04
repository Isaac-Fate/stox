"""This is a command line app based on the framework
provided by the module `click`.
"""

import click
import pandas as pd
import os
import joblib
import json
from datetime import datetime, timedelta

# the following are our modules
import lstm
import strat
import profit
import news



DATA_DIR = "./data"
MODELS_DIR = "./models"
NEWS_DATABASE = os.path.join(DATA_DIR, "news.db")
STOCKS_FILEPATH = os.path.join(DATA_DIR, "stocks.csv")
COMPANIES_FILEPATH = os.path.join(DATA_DIR, "companies.json")
NEWS_DIR_NAME = "news"
NUM_DAYS_LEFT_OUT = 90
    
    
@click.group()
def cli():
    pass



@cli.command()
@click.option("-c", "--company", required=True, help="Company stock/ticker symbol.")
@click.option("-l", "--pred-len", default=1, help="Number of days to predict.", show_default=True)
def train(company, pred_len,):
    """Train an LSTM model to predict future stock prices for a company.
    Model will be saved as ./models/<company ticker>-<prediction length>-day-predictor.pkl.
    For example, ./models/TSLA-3-day-predictor.pkl.
    """
    
    try:
        stocks = pd.read_csv(STOCKS_FILEPATH, index_col=0, parse_dates=True)
    except:
        click.echo("Failed to read stocks data.")
        return
    
    click.echo("Training started...")
    
    try:
        train_result = lstm.train(
            stocks,
            company,
            num_days_left_out=NUM_DAYS_LEFT_OUT,
            pred_len=pred_len
        )
    except:
        click.echo("Failed to train the model.")
        return
    
    model = train_result["model"]
    model_filename = f"{company}-{pred_len}-day-predictor.pkl"
    model_filepath = os.path.join(MODELS_DIR, model_filename)
    
    try:
        joblib.dump(model, model_filepath)
    except:
        click.echo("Failed to save the model. The provided model directory may not exist.")
        return
    
    click.echo(f"Training complete!\nModel is saved as {model_filepath}.")
    
    
    
@cli.command()
@click.option("-c", "--capital", default=10000.0, help="Ammount of money to invest.", show_default=True)  
@click.option("-n", "--last-n-days", default=90, help="The number of days to test.", show_default=True)    
def report_profit(capital, last_n_days):
    """Report the profits using SMA strategy and prediction-based strategy (LSTM), respectively.
    """
    
    num_days_left_out = last_n_days
    
    try:
        stocks = pd.read_csv(STOCKS_FILEPATH, index_col=0, parse_dates=True)
        companies: list[str] = json.load(open(COMPANIES_FILEPATH, "r"))
    except:
        click.echo("Failed to read stocks data or companies.")
        return
    
    
    
    # a summary of profits
    profit_report = {}
    
    for company in companies:
        
        profit_dict = {}
        
        # select stock
        stock = stocks.query(f"Company == '{company}'").drop(columns=["Company", "Sector"])
        
        # load model
        try:
            model_filepath = os.path.join(MODELS_DIR, f"{company}-3-day-predictor.pkl")
            price_predictor = lstm.load_price_predictor(model_filepath)
        except:
            click.echo(f"Failed to load the model {model_filepath}.")
            return
        
        # test data
        stock_test = stock[-num_days_left_out:]
        start_date = stock_test.index[0]
        
        # decide when to buy and sell using SMA
        buy_dates, sell_dates = strat.sma(
            stock,long_period=20
        )
        
        # profit rate of SMA
        profit_rate = profit.calc_profit(stock, buy_dates, sell_dates, start_date)
        profit_dict["SMA"] = profit_rate
        
        # decide when to buy and sell using LSTM
        buy_dates, sell_dates = strat.trade_by_pred(
            stock_test, 
            price_predictor, 
            num_days_ahead=2,
        )
        
        # profit rate of LSTM
        profit_rate = profit.calc_profit(stock, buy_dates, sell_dates, start_date)
        profit_dict["LSTM"] = profit_rate
        
        # add to report
        profit_report[company] = profit_dict
    
    # convert to data frame
    profit_report = pd.DataFrame(profit_report)
    
    # find weights
    weight = strat.determine_portfolio(stocks, companies, num_days_left_out, random_seed=7008)
    
    # weighted profit rates 
    profit_report.loc["Weight"] = weight
    profit_report.loc["Weighted SMA"] = profit_report.loc["Weight"] * profit_report.loc["SMA"]
    profit_report.loc["Weighted LSTM"] = profit_report.loc["Weight"] * profit_report.loc["LSTM"]
    
    # report
    
    pd.options.display.float_format = "{:.2%}".format
    click.echo("Report:")
    print("----------------------------------------------------------------")
    print(profit_report)
    print("----------------------------------------------------------------")
    
    total = profit_report.sum(1).loc[["Weighted SMA", "Weighted LSTM"]] * capital
    total_sma = total["Weighted SMA"]
    total_lstm = total["Weighted LSTM"]
    click.echo(f"Total profit using SMA: {total_sma:.2f}")
    click.echo(f"Total profit using LSTM: {total_lstm:.2f}")
    print("================================================================")
      
        

@cli.command(name="find-company-news")
@click.option("-c", "--company", required=True, help="Company stock/ticker symbol.")
@click.option("-q", "--query", required=True, help="Search query for the news.")
@click.option("-a", "--from-date", default=None, help="Starting date.")
@click.option("-b", "--to-date", default=None, help="Ending date.")
@click.option("-s", "--skip", default=0, show_default=True, help="Number of links to skip.")
@click.option("-f", "--force", is_flag=True, default=False, show_default=True, help="Whether to find headline even if news data exists.")
def find_news_headlines_for_company(company, query, from_date, to_date, skip, force):
    """Find news headlines for the specified company. 
    
    Notes: Each news headline will be inserted into the SQLite database located at ./data/news.db.
    """
    
    try:
        now = datetime.now()
        if from_date is None:
            from_date = now
        else:
            from_date = datetime.strptime(from_date, "%Y-%m-%d")
        if to_date is None:
            to_date = now
        else:
            to_date = datetime.strptime(to_date, "%Y-%m-%d")
    except:
        click.echo("Wrong date format.")
        return
    
    click.echo("Start searching for news headlines...")
    for date in [from_date + timedelta(days=n) for n in range((to_date-from_date).days + 1)]:

        # format date string
        date_str = datetime.strftime(date, "%Y-%m-%d")
        
        # if the news headline already exists,
        # then don't find it again
        if not force and news.fetch_news_headline(NEWS_DATABASE, company, date_str):
            continue
            
        # search and insert news headline to database
        headline = news.search_and_insert_news_healine_to_db(NEWS_DATABASE, company, query, date_str, skip)
        
        # news.search_and_insert_news_healine_to_db("", company, query, date_str, skip)
        click.echo(f"News on {date_str}: {headline}")
    
    click.echo(f"Successfully found all the news for {company}.")
    
    

if __name__ == '__main__':
    cli()