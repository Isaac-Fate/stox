import click
import pandas as pd
import os
import joblib
import json
from datetime import datetime, timedelta

import lstm
import news

DATA_DIR = "./data"
STOCKS_FILEPATH = "./data/stocks.csv"
NEWS_DIR_NAME = "news"

@click.group()
def cli():
    pass

@cli.command()
@click.option("-s", "--stocks", default=STOCKS_FILEPATH, help="Path to the stocks data.", show_default=True)
@click.option("-c", "--company", required=True, help="Company stock/ticker symbol.")
@click.option("-l", "--pred-len", default=1, help="Number of days to predict.", show_default=True)
@click.option("-m", "--model-dir", default="./models", help="Where the model is saved.", show_default=True)
def train(stocks, company, pred_len, model_dir):
    """Train an LSTM model to predict future stock prices for a company.
    """
    
    try:
        stocks_path = stocks
        STOCKS_FILEPATH = stocks_path
        stocks = pd.read_csv(stocks_path, index_col=0, parse_dates=True)
    except:
        click.echo("Failed to read stocks data.")
        return
    
    click.echo("Training started...")
    
    try:
        train_result = lstm.train(
            stocks,
            company,
            pred_len=pred_len
        )
    except:
        click.echo("Failed to train the model.")
        return
    
    model = train_result["model"]
    model_filename = f"{company}-{pred_len}-day-predictor.pkl"
    model_filepath = os.path.join(model_dir, model_filename)
    
    try:
        joblib.dump(model, model_filepath)
    except:
        click.echo("Failed to save the model. The provided model directory may not exist.")
        return
    
    click.echo(f"Training complete!\nModel is saved as {model_filepath}.")

@cli.command(name="find-company-news")
@click.option("-c", "--company", required=True, help="Company stock/ticker symbol.")
@click.option("-q", "--query", required=True, help="Search query for the news.")
@click.option("-a", "--from-date", default=None, help="Starting date.")
@click.option("-b", "--to-date", default=None, help="Ending date.")
@click.option("-s", "--skip", default=0, show_default=True, help="Number of links to skip.")
@click.option("-f", "--force", is_flag=True, default=False, show_default=True, help="Whether to find headline even if news data exists.")
def find_news_headlines_for_company(company, query, from_date, to_date, skip, force):
    """Find news headlines for the specified company. 
    
    Notes: Each news headline will be saved as a single JSON file under the directory ./data/news/<company ticker>/. And the file will be named by the date.
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
    
    # get news dir
    news_dir = os.path.join(DATA_DIR, NEWS_DIR_NAME)
    news_dir = os.path.join(news_dir, company)
    if not os.path.exists(news_dir):
        os.makedirs(news_dir)
    
    click.echo("Start finding news headlines...")
    for date in [from_date + timedelta(days=n) for n in range((to_date-from_date).days + 1)]:

        # format date string
        date_str = datetime.strftime(date, "%Y-%m-%d")
        
        # if the news headline already exists,
        # then don't find it again
        news_filepath = os.path.join(news_dir, f"{date_str}.json")
        if not force and os.path.exists(news_filepath):
            continue
        
        # find headline
        headline = news.find_news_headline(query, date_str, skip)
        click.echo(f"News on {date_str}: {headline}")
        
        # save data
        with open(news_filepath, "w") as f:
            json.dump({"Date": date_str, "Headline": headline}, f, indent=4)
    
    click.echo(f"Successfully found all the news for {company}.")
    
    

if __name__ == '__main__':
    cli()