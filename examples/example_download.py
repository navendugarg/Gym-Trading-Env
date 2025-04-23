import sys  
sys.path.append("./src")


from gym_trading_env.downloader import download, EXCHANGE_LIMIT_RATES
import datetime

EXCHANGE_LIMIT_RATES["bybit"] = {
    "limit":200, # One request will query 200 data points (aka candlesticks) - Adjusted based on Bybit v5 API limits
    "pause_every": 1, # Bybit v5 has stricter rate limits, pause more often
    "pause" : 1, # Pause duration
}
download(
    exchange_names = ["bybit"],
    symbols= ["BTC/USDT", "ETH/USDT"],
    timeframe= "1h",
    data_dir = "examples/data", # Changed 'dir' to 'data_dir'
    since= datetime.datetime(year= 2023, month= 1, day=1),
)