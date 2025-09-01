# common library
import pandas as pd
import numpy as np
import time
import os
from stable_baselines3.common.vec_env import DummyVecEnv
# preprocessor
from preprocess import *
# config
from config.config import *
# model
from model.models import *
def run_model() -> None:
    """Train the model."""
    os.makedirs("visenet/results", exist_ok=True)
    # read and preprocess data
    file_path = "visenet/data/output/top_30_stocks_after_train.csv"
    df = pd.read_csv(file_path)
    feature_cols = ['open','high','low','close','vol','liq','rsi','macd','cci','adx','turbulence']
    data = preprocess_top30(df, feature_cols, top_n=30)
    # 2024/01/01 is the date that validation starts
    # 2025/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    data['timestamp'] = data['timestamp'].astype(int)

    unique_trade_date = data[(data.timestamp > 20240101)&(data.timestamp <= 20250101)].timestamp.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 12
    validation_window = 12

    ## Ensemble Strategy
    run_ensemble_strategy(df=data,
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
