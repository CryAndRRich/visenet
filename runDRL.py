import pandas as pd
import os

from preprocess import *
from config.config import *
from models.ensemble import *

def run_model() -> None:
    """Huấn luyện và đánh giá mô hình DRL trên tập dữ liệu chứng khoán"""
    os.makedirs("visenet/results", exist_ok=True)
    file_path = "visenet/data/output/top_30_stocks_after_train.csv"
    data = pd.read_csv(file_path)

    data["timestamp"] = data["timestamp"].astype(int)

    unique_trade_date = data[(data.timestamp > 20240101)&(data.timestamp <= 20250829)].timestamp.unique()
    print(unique_trade_date)

    rebalance_window = 12
    validation_window = 12

    run_ensemble_strategy(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
