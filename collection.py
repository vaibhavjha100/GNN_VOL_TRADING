import yfinance as yf
import pandas as pd

tickers = [
    # Markets and major
    "SPY", "QQQ", "IWM", "MDY", "^VIX", "^GSPC", "^DJI", "^IXIC", "DX-Y.NYB", "^NYA",
    # Rates & bonds
    "^IRX", "^FVX", "^TNX", "^TYX", "TLT", "IEF", "SHY", "TIP",
    # S&P Sector ETFs
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLU", "XLB", "XLC", "XLRE",
    # Commodities
    "CL=F", "GC=F", "HG=F", "SI=F", "NG=F", "ZB=F", "ZN=F", "USO", "GLD", "SLV", "DBC",
    # Airline & Transports
    "DAL", "UAL", "AAL", "LUV", "FDX", "UPS",
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "NFLX", "AMD", "SMCI", "ASML",
    # Banks
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "EQT", "KMI", "TRP",
    # Healthcare
    "UNH", "LLY", "JNJ", "ABBV", "PFE", "MRNA", "BMY", "VRTX",
    # Global Indices
    "^N225", "^GDAXI", "^FTSE", "000001.SS", "EWJ", "EWG", "EWU", "FXI", "EEM", "EFA", "ACWX",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "^COIN", "MSTR", "GBTC", "ARKK",
    # Materials
    "FCX", "CAT", "DE", "BA", "HON", "GE", "NEM"
]

data = pd.DataFrame()

for ticker in tickers:
    sdf = yf.download(ticker, start="2015-01-01", interval="1d", multi_level_index=False)
    sdf["Ticker"] = ticker
    data = pd.concat([data,sdf])

# Drop Adjusted Close if exists
if 'Adj Close' in data.columns:
    data = data.drop(columns=['Adj Close'])

# Set multi-index
data.index.name = 'Date'
data = data.set_index('Ticker', append=True)
data = data.reorder_levels(['Ticker', 'Date'])


data.to_csv("data.csv")

print("Data downloaded successfully")
print(data.head())
print(data.info())
