import pandas as pd
import numpy as np

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
