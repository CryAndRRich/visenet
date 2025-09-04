# !pip install pandas_market_calendars

import numpy as np

import pandas as pd
import pandas_market_calendars as mcal

import matplotlib.pyplot as plt
from datetime import datetime

def analyze_portfolio(df: pd.DataFrame) -> tuple:
    """
    Phân tích hiệu suất danh mục đầu tư. Trả về các chỉ số:
    - Cumulative return
    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Max drawdown
    - Win rate
    """
    account_value = df["account_value"].reset_index(drop=True)
    
    # Lãi hàng ngày
    returns = account_value.pct_change().dropna()
    N = len(returns)  # số ngày
    
    # Cumulative return
    cumulative_return = (account_value.iloc[-1] / account_value.iloc[0]) - 1
    
    # Annualized return
    annualized_return = (account_value.iloc[-1] / account_value.iloc[0]) ** (252 / N) - 1 if N > 0 else np.nan
    
    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(252) if N > 0 else np.nan
    
    # Sharpe ratio
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan
    
    # Max drawdown
    cummax = account_value.cummax()
    drawdown = (account_value - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Win rate (tỷ lệ số ngày return > 0)
    win_rate = (returns > 0).mean() if N > 0 else np.nan

    return cumulative_return, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, win_rate

def add_holiday_hold(df, start_date="2016-01-04", end_date=None, market="XNYS"):
    """
    Chỉnh sửa DataFrame để bao gồm cả ngày lễ và cuối tuần.
    Các ngày không giao dịch sẽ được điền giá trị bằng phương pháp forward fill
    """
    if end_date is None:
        end_date = pd.to_datetime(start_date) + pd.offsets.BDay(len(df)-1)
    
    # lịch thị trường
    cal = mcal.get_calendar(market)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index  # các ngày thực sự có giao dịch
    
    # gán index = trading days
    df = df.copy()
    df.index = trading_days
    
    # tạo lịch ngày thường (daily)
    full_range = pd.date_range(start=trading_days[0], end=trading_days[-1], freq="D")
    
    # reindex và forward fill (bao gồm cả weekend + holiday)
    df = df.reindex(full_range, method="ffill")
    df.index.name = "date"
    return df

def plot_performance() -> None:
    """
    Vẽ biểu đồ hiệu suất danh mục đầu tư
    """
    # Danh sách file và label tương ứng
    files = {
        "A2C": "a2c_account_value_trade.csv",
        "PPO": "ppo_account_value_trade.csv",
        "TD3": "td3_account_value_trade.csv",
        "VISENET": "ensemble_account_value_trade.csv",
        "Min-Variance": "minvar_account_value_trade.csv",
        "VN-INDEX": "benchmark_vnindex.csv"
    }

    # Ngày bắt đầu và kết thúc
    start_date = datetime(2024, 4, 5)

    plt.figure(figsize=(12, 6))

    for label, path in files.items():
        df = pd.read_csv(path)
        print(label)
        df = add_holiday_hold(df, start_date="2024-04-05", end_date="2025-07-14")
        n_days = df.shape[0]
        df['date'] = pd.date_range(start=start_date, periods=n_days, freq='D')
        df['cumulative_return'] = df['account_value'] / df['account_value'].iloc[0] - 1
        plt.plot(df['date'], df['cumulative_return'], label=label)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_return.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_drawdown() -> None:
    files = {
        "VISENET": "ensemble_account_value_trade.csv"
    }

    start_date = datetime(2024, 4, 5)

    # Load cumulative return
    all_returns = []
    for label, path in files.items():
        df = pd.read_csv(path)
        df = add_holiday_hold(df, start_date="2024-04-05", end_date="2025-07-9")
        n_days = df.shape[0]
        df["date"] = pd.date_range(start=start_date, periods=n_days, freq="D")
        df[f"cumulative_return_{label}"] = df["account_value"] / df["account_value"].iloc[0] - 1
        all_returns.append(df[["date", f"cumulative_return_{label}"]])

    # Merge vào 1 dataframe chung
    merged = all_returns[0]
    for df in all_returns[1:]:
        merged = merged.merge(df, on="date")

    returns_df = merged.set_index("date")

    # Tính drawdown từ trung bình
    avg_return = returns_df.mean(axis=1)
    cummax = avg_return.cummax()
    drawdown = avg_return / cummax - 1

    # Xác định giai đoạn drawdown
    periods = []
    in_dd, peak, trough = False, None, None

    for i in range(len(drawdown)):
        if not in_dd:
            if drawdown.iloc[i] < 0:
                in_dd, peak, trough = True, i, i
        else:
            if drawdown.iloc[i] < drawdown.iloc[trough]:
                trough = i
            # khi drawdown tăng lại (giảm bớt âm), coi như kết thúc 1 giai đoạn giảm
            if drawdown.iloc[i] > drawdown.iloc[i-1]:
                periods.append((peak, trough, i))
                in_dd = False

    # Nếu cuối cùng vẫn đang giảm thì cũng tính
    if in_dd:
        periods.append((peak, trough, len(drawdown)-1))

    # Lọc top theo mức giảm mạnh nhất (slope lớn nhất về âm)
    def drop_slope(p):
        peak, trough, _ = p
        dd_change = drawdown.iloc[trough] - drawdown.iloc[peak]
        days = trough - peak if trough > peak else 1
        return dd_change / days  # slope

    periods_sorted = sorted(periods, key=lambda x: drop_slope(x))[:8]

    plt.figure(figsize=(14, 7))
    colors = {"VISENET": "red"}

    for label in files.keys():
        plt.plot(returns_df.index, returns_df[f"cumulative_return_{label}"],
                label=label, color=colors[label])

    # Highlight vùng giảm mạnh
    for peak, trough, recovery in periods_sorted:
        plt.axvspan(drawdown.index[peak], drawdown.index[recovery],
                    color="grey", alpha=0.3)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.xlim(datetime(2025, 2, 1), datetime(2025, 6, 1))
    plt.tight_layout()
    plt.savefig("top_3_drawdown.png", dpi=300, bbox_inches='tight')
    plt.show()