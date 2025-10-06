import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

# =========================
# Calculate monthly factors
# =========================
def month_end_factors(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly factors
    
    Parameters:
        prices_df: DataFrame with columns ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    
    Returns:
        monthly: DataFrame with columns ["timestamp", "ticker", "rsi", "vol", "liq", "close"]
    """
    df = prices_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    monthly = df.groupby(["ticker", pd.Grouper(freq="ME")]).last().reset_index()
    return monthly

# =========================
# Ranking and calculating portfolio returns
# =========================
def rank_0_1(series: pd.Series, 
             reverse=False) -> pd.Series:
    """
    Rank the series to [0, 1]
    
    Parameters:
        series: pd.Series
        reverse: bool, if True, rank in descending order
        
    Returns:
        pd.Series with ranks in [0, 1]
    """
    if reverse:
        series = -series
    ranks = series.rank(method="average", pct=True)
    return ranks.fillna(0.5)

def portfolio_returns(weights: tuple, 
                      factors: dict, 
                      price_monthly: pd.DataFrame, 
                      shares_map: pd.Series, 
                      top_pct: float = 0.1) -> pd.DataFrame:
    """
    Calculate portfolio returns based on weighted factors
    
    Parameters:
        weights: tuple of weights for (MOM, VOL, LIQ)
        factors: dict with keys "MOM", "VOL", "LIQ" and values as DataFrames
        price_monthly: DataFrame of monthly closing prices (index: month, columns: tickers)
        shares_map: Series mapping ticker to outstanding shares
        top_pct: float, percentage of top stocks to select
    
    Returns:
        df: DataFrame with index as month and columns ["ret", "MV"]
    """
    # Weights for MOM, VOL, LIQ
    w1, w2, w3 = weights
    mom, vol, liq = factors["MOM"], factors["VOL"], factors["LIQ"]
    months = price_monthly.index
    results = []

    for i in range(len(months) - 1):
        m, m_next = months[i], months[i + 1]

        if m not in mom.index:
            continue

        s1 = rank_0_1(mom.loc[m])
        s2 = rank_0_1(vol.loc[m], reverse=True) # lower volatility better
        s3 = rank_0_1(liq.loc[m]) # higher liquidity better

        tickers = price_monthly.columns.intersection(s1.index).intersection(s2.index).intersection(s3.index)
        if len(tickers) == 0: 
            continue

        scores = w1 * s1[tickers] + w2 * s2[tickers] + w3 * s3[tickers]
        k = max(1, int(len(scores) * top_pct))
        top = scores.nlargest(k).index

        p0 = price_monthly.loc[m, top]
        p1 = price_monthly.loc[m_next, top]
        mask = p0.notna() & p1.notna()
        if mask.sum() == 0: 
            continue

        ret = (p1[mask] / p0[mask] - 1).mean()

        # median MV = price * shares_outstanding
        mv_vals = (p0[mask] * shares_map[top].reindex(p0[mask].index)).dropna()
        if mv_vals.empty: 
            continue
        mv_med = mv_vals.median()

        results.append((m, ret, mv_med))

    df = pd.DataFrame(results, columns=["month", "ret", "MV"]).set_index("month")
    return df

# =========================
# Estimating alpha, beta
# =========================
def estimate_alpha_beta(port_ret: pd.Series, 
                        market_ret: pd.Series, 
                        rf: float = 0.0) -> tuple:
    """
    Estimate alpha and beta using OLS regression
    
    Parameters:
        port_ret: pd.Series of portfolio returns (index: month)
        market_ret: pd.Series of market returns (index: month)
        rf: risk-free rate, default 0.0
    
    Returns:
        (alpha, beta): tuple of estimated alpha and beta
    """
    idx = port_ret.index.intersection(market_ret.index)
    y = port_ret.loc[idx] - rf
    x = market_ret.loc[idx] - rf
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit() #OLS regression
    alpha, beta = model.params
    return alpha, beta


# =========================
# Fit regression
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for regression model
    
    Parameters:
        df: DataFrame with columns ["w1", "w2", "w3"]

    Returns:
        X: DataFrame with polynomial features and interaction terms
    """
    W = df[["w1", "w2", "w3"]].values
    w1, w2, w3 = W[:, 0], W[:,1], W[:,2]
    X = np.column_stack([
        w1, w2, w3,
        w1 * w2, w2 * w3, w3 * w1,
        w1 * w2 * w3
    ])
    return sm.add_constant(X)

# =========================
# Optimize weights
# =========================
def predict(models: dict, 
            w: tuple) -> tuple:
    """
    Predict alpha, beta, mv using regression models
    
    Parameters:
        models: dict with keys "alpha", "beta", "mv" and values as fitted OLS models
        w: tuple of weights (w1, w2, w3)
    
    Returns:
        (f_alpha, f_beta, f_mv): tuple of predicted values
    """
    w1, w2, w3 = w

    feats = [1,
             w1, w2, w3,
             w1 * w2, w2 * w3, w3 * w1,
             w1 * w2 * w3]
    
    f_alpha = np.dot(models["alpha"].params, feats)
    f_beta = np.dot(models["beta"].params, feats)
    f_mv = np.dot(models["mv"].params, feats)
    return f_alpha, f_beta, f_mv

def objective(w: tuple) -> float:
    """
    Objective function to minimize (negative alpha)
    
    Parameters:
        w: tuple of weights (w1, w2, w3)
        
    Returns:
        -f_alpha: negative predicted alpha"""
    f_alpha, _, _ = predict(models, w)
    return -f_alpha


# =========================
# Pick top 30 stocks based on optimal weights
# =========================
def get_top_scores(weights: tuple, 
                   factors: dict, 
                   top: int = 30) -> pd.DataFrame:
    """
    Get top stocks based on weighted factor scores
    
    Parameters:
        weights: tuple of weights (w1, w2, w3)
        factors: dict with keys "MOM", "VOL", "LIQ" and values as DataFrames
        top: int, number of top stocks to select
    
    Returns:
        top_stocks_df: DataFrame with columns ["ticker", "rsi_score", "vol_score", "liq_score", "overall_score"]
    """
    w1, w2, w3 = weights
    mom_df, vol_df, liq_df = factors["MOM"], factors["VOL"], factors["LIQ"]

    last_month = mom_df.index[-1]
    mom_series = mom_df.loc[last_month]
    vol_series = vol_df.loc[last_month]
    liq_series = liq_df.loc[last_month]

    s1 = rank_0_1(mom_series)
    s2 = rank_0_1(vol_series, reverse=True)
    s3 = rank_0_1(liq_series)

    scores = w1 * s1 + w2 * s2 + w3 * s3

    output_df = pd.DataFrame({
        "rsi_score": (s1 * 100).round(2),
        "vol_score": (s2 * 100).round(2),
        "liq_score": (s3 * 100).round(2),
        "overall_score": (scores * 100).round(2)
    }).reset_index(names="ticker")

    output_df = output_df.sort_values(by="overall_score", ascending=False)
    k = max(1, top)
    top_stocks_df = output_df.head(k).reset_index(drop=True)
    return top_stocks_df


def preprocess_top30(df: pd.DataFrame, 
                     top_n: int = 30) -> pd.DataFrame:
    """
    Ensure each month has exactly top_n stocks, filling from nearest months if needed
    
    Parameters:
        df: DataFrame with columns ["timestamp", "ticker", ...]
        top_n: int, number of stocks per month

    Returns:
        df_out: DataFrame with exactly top_n stocks per month
    """
    result = []
    all_dates = sorted(df["timestamp"].unique())
    
    for i, date in enumerate(all_dates):
        top_df = df[df["timestamp"] == date]
        
        # If already have enough stocks
        if len(top_df) == top_n:
            result.append(top_df)
        else:
            # Need to fill
            missing = top_n - len(top_df)
            j = i - 1
            filled = []
            while j >= 0 and len(filled) < missing:
                prev_day = result[j] 
                # Take tickers not already in top_df
                candidates = prev_day[~prev_day["ticker"].isin(top_df["ticker"])]
                needed = candidates.head(missing - len(filled))
                filled.append(needed)
                j -= 1
            if len(filled) < missing:
                k = i + 1
                while k < len(all_dates) and len(filled) < missing:
                    next_day = df[df["timestamp"] == all_dates[k]].nlargest(top_n, "vol")
                    candidates = next_day[~next_day["ticker"].isin(top_df["ticker"])]
                    needed = candidates.head(missing - len(filled))
                    filled.append(needed)
                    k += 1

            # Merge filled data
            filled_df = pd.concat(filled) if filled else pd.DataFrame(columns=top_df.columns)
            final_day = pd.concat([top_df, filled_df]).head(top_n)
            final_day["timestamp"] = date  # Make sure timestamp is correct
            result.append(final_day)
    
    df_out = pd.concat(result).sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return df_out


# =========================
# Calculate turbulence index
# =========================
def add_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add turbulence index to the DataFrame
    
    Parameters:
        df: DataFrame with columns ["timestamp", "ticker", "close", ...]
    
    Returns:
        df: DataFrame with added "turbulence" column
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on="timestamp")
    df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return df


def calcualte_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate turbulence index based on Mahalanobis distance
    
    Parameters:
        df: DataFrame with columns ["timestamp", "ticker", "close"]
    
    Returns:
        turbulence_index: DataFrame with columns ["timestamp", "turbulence"]
    """
    df_price_pivot = df.pivot(index="timestamp", columns="ticker", values="close")
    unique_date = df.timestamp.unique()

    # Start calculating turbulence from the 252nd day (1 year of trading days)
    start = 252
    turbulence_index = [0] * start
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # Avoid large turbulence in the beginning
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({"timestamp": df_price_pivot.index,
                                     "turbulence": turbulence_index})
    return turbulence_index


if __name__ == "__main__":
    # =========================
    # Load data
    # =========================
    prices_df = pd.read_csv("clean_data_1029_tickers_29_11_2018_to_29_8_2025.csv", parse_dates=["timestamp"])
    shares_df = pd.read_csv("outstanding_shares.csv")

    # Training set
    train_end_date = "2023-12-31"
    prices_train_df = prices_df[prices_df["timestamp"] <= train_end_date].copy()

    # Keep only tickers that appear in both datasets
    common_tickers = set(prices_train_df["ticker"]).intersection(set(shares_df["ticker"]))
    prices_train_df = prices_train_df[prices_train_df["ticker"].isin(common_tickers)]
    shares_df = shares_df[shares_df["ticker"].isin(common_tickers)]

    # Testing set
    test_start_date = "2024-01-01"
    prices_test_df = prices_df[prices_df["timestamp"] >= test_start_date].copy()
    prices_test_df = prices_test_df[prices_test_df["ticker"].isin(common_tickers)]

    monthly_factors_train = month_end_factors(prices_train_df)

    # Pivot to wide format
    momentum_df_train = monthly_factors_train.pivot(index="timestamp", columns="ticker", values="rsi")
    volatility_df_train = monthly_factors_train.pivot(index="timestamp", columns="ticker", values="vol")
    liquidity_df_train = monthly_factors_train.pivot(index="timestamp", columns="ticker", values="liq")
    close_df_train = monthly_factors_train.pivot(index="timestamp", columns="ticker", values="close")

    factors_monthly_train = {
        "MOM": momentum_df_train, 
        "VOL": volatility_df_train, 
        "LIQ": liquidity_df_train
    }

    # =========================
    # Monthly market returns
    # =========================
    returns_monthly_train = close_df_train.pct_change(fill_method=None).dropna(how="all")
    market_ret_train = returns_monthly_train.mean(axis=1)

    # Number of outstanding shares
    shares_map = shares_df.set_index("ticker")["outstanding_share"]

    # =========================
    # Model with different weight combinations
    # =========================
    mixes = [
        (1, 0, 0), (0,1, 0), (0, 0,1),
        (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2),
        (1/3, 1/3, 1/3)
    ]

    results = []
    for w in mixes:
        dfp = portfolio_returns(w, factors_monthly_train, close_df_train, shares_map)
        if dfp.empty: 
            continue
        alpha, beta = estimate_alpha_beta(dfp["ret"], market_ret_train)
        mv_stat = np.log(dfp["MV"]).mean()

        results.append({
            "w1": w[0],
            "w2": w[1],
            "w3": w[2],
            "alpha": alpha, 
            "beta": beta, 
            "logMV": mv_stat
        })

    results_df = pd.DataFrame(results)
    print("Regression Data:\n", results_df)

    X = build_features(results_df)
    y_alpha, y_beta, y_mv = results_df["alpha"], results_df["beta"], results_df["logMV"]

    model_alpha = sm.OLS(y_alpha, X).fit()
    model_beta = sm.OLS(y_beta, X).fit()
    model_mv = sm.OLS(y_mv, X).fit()

    models = {
        "alpha": model_alpha, 
        "beta": model_beta, 
        "mv": model_mv
    }

    beta_star = results_df["beta"].median()
    mv_star = results_df["logMV"].median()

    cons = [
        {
            "type": "eq", 
            "fun": lambda w: np.sum(w) - 1
        }, {
            "type": "ineq", 
            "fun": lambda w: beta_star - predict(models, w)[1]
        }, {
            "type": "ineq", 
            "fun": lambda w: predict(models, w)[2] - mv_star
        }
    ]
    bnds = [(0, 1)] * 3

    res = minimize(objective, x0=[1/3, 1/3, 1/3], bounds=bnds, constraints=cons)
    print("Optimal weights:", res.x)

    optimal_weights = res.x

    top_score_df = get_top_scores(optimal_weights, factors_monthly_train)
    top_score_df.to_csv("top_30_score_after_train.csv", index = False)
    
    top_stocks_df = prices_df[prices_df["ticker"].isin(top_score_df["ticker"])]

    # Change datetime format
    top_stocks_df["timestamp"] = pd.to_datetime(top_stocks_df["timestamp"])
    top_stocks_df["timestamp"] = top_stocks_df["timestamp"].dt.strftime("%Y%m%d")
    
    # Add turbulence
    top_stocks_df = add_turbulence(top_stocks_df)

    top_stocks_df = preprocess_top30(top_stocks_df, top_n=30)
    top_stocks_df.to_csv("top_30_stocks_after_train.csv", index = False)