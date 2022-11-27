import click
import pandas as pd
import os
import joblib
import pickle

import lstm

@click.group()
def cli():
    pass

@cli.command()
@click.option("-s", "--stocks", default="./data/stocks.csv", help="Path to the stocks data.", show_default=True)
@click.option("-c", "--company", required=True, help="Company stock/ticker symbol.")
@click.option("-l", "--pred-len", default=1, help="Number of days to predict.", show_default=True)
@click.option("-m", "--model-dir", default="./models", help="Where the model is saved.", show_default=True)
def train(stocks, company, pred_len, model_dir):
    """Train an LSTM model to predict future stock prices for a company.
    """
    
    try:
        stocks_path = stocks
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
    
    click.echo(f"Training complete! Model is saved as {model_filepath}.")

if __name__ == '__main__':
    cli()