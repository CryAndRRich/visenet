import pandas as pd

def preprocess_top30(df, feature_cols, top_n=30):
    """
    Chuẩn hóa data sao cho mỗi ngày có đúng top_n tickers.
    Nếu ngày nào thiếu thì fill từ ngày gần nhất (trước hoặc sau).
    
    df: DataFrame có cột ['ticker', 'timestamp', ... feature_cols ...]
    feature_cols: list tên các cột feature (open, high, low, close, ...).
    top_n: số lượng tickers cần giữ lại mỗi ngày.
    """
    result = []
    all_dates = sorted(df['timestamp'].unique())
    
    for i, date in enumerate(all_dates):
        day_df = df[df['timestamp'] == date]
        
        # Lấy top_n tickers (ví dụ dựa vào volume)
        top_df = day_df.nlargest(top_n, 'vol')
        
        # Nếu đủ top_n thì ok
        if len(top_df) == top_n:
            result.append(top_df)
        else:
            # Cần fill thêm
            missing = top_n - len(top_df)
            # Tìm ngày gần nhất có data đủ
            j = i - 1
            filled = []
            while j >= 0 and len(filled) < missing:
                prev_day = result[j]  # đã được chuẩn hóa từ trước
                # lấy ticker chưa có trong ngày hiện tại
                candidates = prev_day[~prev_day['ticker'].isin(top_df['ticker'])]
                needed = candidates.head(missing - len(filled))
                filled.append(needed)
                j -= 1
            # nếu vẫn chưa đủ thì lấy từ ngày sau
            if len(filled) < missing:
                k = i + 1
                while k < len(all_dates) and len(filled) < missing:
                    next_day = df[df['timestamp'] == all_dates[k]].nlargest(top_n, 'vol')
                    candidates = next_day[~next_day['ticker'].isin(top_df['ticker'])]
                    needed = candidates.head(missing - len(filled))
                    filled.append(needed)
                    k += 1
            # gộp lại
            filled_df = pd.concat(filled) if filled else pd.DataFrame(columns=day_df.columns)
            final_day = pd.concat([top_df, filled_df]).head(top_n)
            final_day['timestamp'] = date  # đảm bảo timestamp đúng
            result.append(final_day)
    
    df_out = pd.concat(result).sort_values(['timestamp', 'ticker']).reset_index(drop=True)
    df_out.to_csv("top_30_stocks_after_train_processed", index=False)
    return df_out
file_path = "visenet/data/output/top_30_stocks_after_train.csv"
df = pd.read_csv(file_path)
feature_cols = ['open','high','low','close','vol','liq','rsi','macd','cci','adx','turbulence']
data_fixed = preprocess_top30(df, feature_cols, top_n=30)
