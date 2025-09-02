# !pip install --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
# !pip install --upgrade --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx

import pandas as pd
import numpy as np

from FiinQuantX import FiinSession

# Đăng nhập vào FiinQuant
username = "USERNAME"  
password = "PASSWORD"

client = FiinSession(username=username, password=password).login()
fi = client.FiinIndicator()

def compute_indicators(grp: pd.DataFrame) -> pd.DataFrame:
    """Tính các chỉ số kỹ thuật cho từng nhóm mã chứng khoán"""
    grp = grp.reset_index(drop=True)

    # Giá đóng cửa phiên trước
    grp["prev_close"] = grp["close"].shift(1)

    # Log Return
    grp["log_return"] = np.log(grp["close"] / grp["prev_close"])

    # Volatility 20 ngày annualized
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

def data_split(df, start, end):
    """Tách dữ liệu thành tập huấn luyện hoặc kiểm tra dựa trên ngày tháng"""
    data = df[(df.timestamp >= start) & (df.timestamp < end)]
    data=data.sort_values(['timestamp', 'ticker'], ignore_index=True)
    data.index = data.timestamp.factorize()[0]
    return data


if __name__ == "__main__":
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('data/raw_data_all_tickers_1d_30_8_2018_to_30_8_2025.csv')

    # Áp dụng hàm tính chỉ số kỹ thuật cho từng mã chứng khoán
    data = data.groupby("ticker", group_keys=False).apply(compute_indicators)

    # Loại bỏ các cột không cần thiết và các dòng có giá trị NaN
    data.drop(columns=['volume', 'bu', 'sd', 'fb', 'fs', 'fn', 'log_return', 'prev_close', 'tr'], inplace = True)
    data = data.dropna()

    # Lọc các mã chứng khoán có ít nhất 1500 ngày giao dịch
    counts = data['ticker'].value_counts()
    data = data[data['ticker'].isin(counts[counts >= 1500].index)]

    # Lưu dữ liệu vào file CSV
    data.to_csv("data/clean_data_1029_tickers_29_11_2018_to_29_8_2025.csv", index=False)