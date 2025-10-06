# !pip install --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
# !pip install --upgrade --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx

import pandas as pd
import numpy as np

from FiinQuantX import FiinSession

# Login into FiinQuantX
username = "USERNAME"  
password = "PASSWORD"

client = FiinSession(username=username, password=password).login()
fi = client.FiinIndicator()

def compute_indicators(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given stock group
    
    Parameters:
        grp: DataFrame containing stock data for a single ticker
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    grp = grp.reset_index(drop=True)

    # Close price of the previous day
    grp["prev_close"] = grp["close"].shift(1)

    # Log Return
    grp["log_return"] = np.log(grp["close"] / grp["prev_close"])

    # Volatility (20-day rolling standard deviation of log returns, annualized)
    grp["vol"] = grp["log_return"].rolling(20, min_periods=1).std() * np.sqrt(252)

    # True Range
    grp["tr"] = np.maximum.reduce([
        grp["high"] - grp["low"],
        (grp["high"] - grp["prev_close"]).abs(),
        (grp["low"] - grp["prev_close"]).abs()
    ])

    # Liquidity (14-day average True Range)
    grp["liq"] = grp["tr"].rolling(14, min_periods=1).mean()

    # RSI
    grp["rsi"] = fi.rsi(grp["close"], window=14)

    # MACD
    grp["macd"] = fi.macd(grp["close"], window_fast=12, window_slow=26)

    # CCI
    grp["cci"] = fi.cci(grp["high"], grp["low"], grp["close"], window=20, constant=0.015)

    # ADX
    grp["adx"] = fi.adx(grp["high"], grp["low"], grp["close"], window=14)

    return grp

def data_split(df: pd.DataFrame, 
               start: str, 
               end : str) -> pd.DataFrame:
    """
    Split the DataFrame based on a date range and reindex it
    
    Parameters:
        df: DataFrame containing stock data
        start: Start date as a string (inclusive)
        end: End date as a string (exclusive)
    
    Returns:
        pd.DataFrame: Filtered and reindexed DataFrame
    """
    data = df[(df.timestamp >= start) & (df.timestamp < end)]
    data=data.sort_values(['timestamp', 'ticker'], ignore_index=True)
    data.index = data.timestamp.factorize()[0]
    return data


if __name__ == "__main__":
    # Read raw data from CSV file
    data = pd.read_csv('data/raw_data_all_tickers_1d_30_8_2018_to_30_8_2025.csv')

    # Apply the function to compute technical indicators for each stock ticker
    data = data.groupby("ticker", group_keys=False).apply(compute_indicators)

    # Delete unnecessary columns and rows with NaN values
    data.drop(columns=['volume', 'bu', 'sd', 'fb', 'fs', 'fn', 'log_return', 'prev_close', 'tr'], inplace = True)
    data = data.dropna()

    # Filter stock tickers with at least 1500 trading days
    counts = data['ticker'].value_counts()
    data = data[data['ticker'].isin(counts[counts >= 1500].index)]

    # Save data to CSV file
    data.to_csv("data/clean_data_1029_tickers_29_11_2018_to_29_8_2025.csv", index=False)