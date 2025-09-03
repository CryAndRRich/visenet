import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

# =========================
# Tính các yếu tố hàng tháng
# =========================
def month_end_factors(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    monthly = df.groupby(["ticker", pd.Grouper(freq="ME")]).last().reset_index()
    return monthly

# =========================
# Sap xếp và tính lợi nhuận danh mục
# =========================
def rank_0_1(series: pd.Series, 
             reverse=False) -> pd.Series:
    if reverse:
        series = -series
    ranks = series.rank(method="average", pct=True)
    return ranks.fillna(0.5)

def portfolio_returns(weights: tuple, 
                      factors: dict, 
                      price_monthly: pd.DataFrame, 
                      shares_map: pd.Series, 
                      top_pct: float = 0.1) -> pd.DataFrame:

    # Trọng số cho MOM, VOL, LIQ
    w1, w2, w3 = weights
    mom, vol, liq = factors["MOM"], factors["VOL"], factors["LIQ"]
    months = price_monthly.index
    results = []

    for i in range(len(months) - 1):
        m, m_next = months[i], months[i + 1]

        if m not in mom.index:
            continue

        s1 = rank_0_1(mom.loc[m])
        s2 = rank_0_1(vol.loc[m], reverse=True) # vol thấp tốt
        s3 = rank_0_1(liq.loc[m]) # liquidity cao tốt

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
# Uớc lượng alpha, beta
# =========================
def estimate_alpha_beta(port_ret: pd.Series, 
                        market_ret: pd.Series, 
                        rf: float = 0.0) -> tuple:
    
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
    W = df[["w1", "w2", "w3"]].values
    w1, w2, w3 = W[:, 0], W[:,1], W[:,2]
    X = np.column_stack([
        w1, w2, w3,
        w1 * w2, w2 * w3, w3 * w1,
        w1 * w2 * w3
    ])
    return sm.add_constant(X)

# =========================
# Tối ưu trọng số
# =========================
def predict(models: dict, 
            w: tuple) -> tuple:
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
    f_alpha, _, _ = predict(models, w)
    return -f_alpha


# =========================
# Chọn top 30 với trọng số tối ưu
# =========================
def get_top_scores(weights: tuple, 
                   factors: dict, 
                   top: int = 30) -> pd.DataFrame:
    
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


def preprocess_top30(df, top_n=30):
    """
    Chuẩn hóa data sao cho mỗi ngày có đúng top_n tickers.
    Nếu ngày nào thiếu thì fill từ ngày gần nhất (trước hoặc sau)
    """
    result = []
    all_dates = sorted(df["timestamp"].unique())
    
    for i, date in enumerate(all_dates):
        top_df = df[df["timestamp"] == date]
        
        # Nếu đủ top_n 
        if len(top_df) == top_n:
            result.append(top_df)
        else:
            # Cần bổ sung thêm
            missing = top_n - len(top_df)
            # Tìm ngày gần nhất có data đủ
            j = i - 1
            filled = []
            while j >= 0 and len(filled) < missing:
                prev_day = result[j]  # đã được chuẩn hóa từ trước
                # lấy ticker chưa có trong ngày hiện tại
                candidates = prev_day[~prev_day["ticker"].isin(top_df["ticker"])]
                needed = candidates.head(missing - len(filled))
                filled.append(needed)
                j -= 1
            # nếu vẫn chưa đủ thì lấy từ ngày sau
            if len(filled) < missing:
                k = i + 1
                while k < len(all_dates) and len(filled) < missing:
                    next_day = df[df["timestamp"] == all_dates[k]].nlargest(top_n, "vol")
                    candidates = next_day[~next_day["ticker"].isin(top_df["ticker"])]
                    needed = candidates.head(missing - len(filled))
                    filled.append(needed)
                    k += 1
            # gộp lại
            filled_df = pd.concat(filled) if filled else pd.DataFrame(columns=top_df.columns)
            final_day = pd.concat([top_df, filled_df]).head(top_n)
            final_day["timestamp"] = date  # đảm bảo timestamp đúng
            result.append(final_day)
    
    df_out = pd.concat(result).sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return df_out


# =========================
# Tính toán turbulence
# =========================
def add_turbulence(df):
    """Thêm chỉ số turbulence"""
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on="timestamp")
    df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return df


def calcualte_turbulence(df):
    """Tính chỉ số turbulence"""
    df_price_pivot = df.pivot(index="timestamp", columns="ticker", values="close")
    unique_date = df.timestamp.unique()

    # Bắt đầu sau một năm
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
                # Tránh outlier lớn
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

    # Tập train
    train_end_date = "2023-12-31"
    prices_train_df = prices_df[prices_df["timestamp"] <= train_end_date].copy()

    # Lọc ticker xuất hiện trong cả hai tập
    common_tickers = set(prices_train_df["ticker"]).intersection(set(shares_df["ticker"]))
    prices_train_df = prices_train_df[prices_train_df["ticker"].isin(common_tickers)]
    shares_df = shares_df[shares_df["ticker"].isin(common_tickers)]

    # Tập test
    test_start_date = "2024-01-01"
    prices_test_df = prices_df[prices_df["timestamp"] >= test_start_date].copy()
    prices_test_df = prices_test_df[prices_test_df["ticker"].isin(common_tickers)]

    monthly_factors_train = month_end_factors(prices_train_df)

    # Pivot các factor
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
    # Lãi suất thị trường hàng tháng
    # =========================
    returns_monthly_train = close_df_train.pct_change(fill_method=None).dropna(how="all")
    market_ret_train = returns_monthly_train.mean(axis=1)

    # Số lượng cổ phiếu lưu hành
    shares_map = shares_df.set_index("ticker")["outstanding_share"]

    # =========================
    # Mô hình với các tổ hợp trọng số khác nhau
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
    print("Optimal weights sau train:", res.x)

    optimal_weights = res.x

    top_score_df = get_top_scores(optimal_weights, factors_monthly_train)
    top_score_df.to_csv("top_30_score_after_train.csv", index = False)
    
    top_stocks_df = prices_df[prices_df["ticker"].isin(top_score_df["ticker"])]

    # Đổi định dạng ngày tháng
    top_stocks_df["timestamp"] = pd.to_datetime(top_stocks_df["timestamp"])
    top_stocks_df["timestamp"] = top_stocks_df["timestamp"].dt.strftime("%Y%m%d")
    
    # Thêm turbulence
    top_stocks_df = add_turbulence(top_stocks_df)

    top_stocks_df = preprocess_top30(top_stocks_df, top_n=30)
    top_stocks_df.to_csv("top_30_stocks_after_train.csv", index = False)