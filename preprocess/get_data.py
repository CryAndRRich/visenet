# !pip install --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
# !pip install --upgrade --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx

from FiinQuantX import FiinSession

# Login into FiinQuantX
username = "USERNAME"  
password = "PASSWORD"

client = FiinSession(username=username, password=password).login()

# Get the list of stock tickers from HOSE, HNX, UPCOM exchanges
hose_list = client.TickerList(ticker="VNINDEX") # HOSE
hnx_list  = client.TickerList(ticker="HNXINDEX") # HNX
upcom_list = client.TickerList(ticker="UPCOMINDEX") # UPCOM

all_tickers = list(set(hose_list + hnx_list + upcom_list))

print("Tổng số mã:", len(all_tickers))
print(all_tickers)

# Fetch daily trading data for all stock tickers
data = client.Fetch_Trading_Data(
    realtime = False,
    tickers = all_tickers,
    fields = ["open", "high", "low", "close", "volume", "bu", "sd", "fb", "fs", "fn"],
    adjusted=True,
    by = "1d",
    from_date="2018-8-30 00:00",
    to_date = "2025-8-30 00:00"
).get_data()

print(data)

# Save data to CSV file
data.to_csv("data/raw_data_all_tickers_1d_30_8_2018_to_30_8_2025.csv", index=False)