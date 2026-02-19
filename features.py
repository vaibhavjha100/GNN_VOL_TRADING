import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data.csv", index_col=[0,1])  # ('Ticker','Date') MultiIndex
df.index = df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
df = df.sort_index(level=1)

unique_tickers = df.index.get_level_values(0).unique()
print(f"Tickers: {unique_tickers[:5]}... ({len(unique_tickers)} total)")

# Node Features: Realized Volatility, Skewness, Kurtosis, Volume_proxy

window_short = 21
window_long = 252

features_list = []  # Better than concat in loop

for ticker in unique_tickers:
    data = df.xs(ticker, level=0).copy()  # .copy() prevents warnings
    
    # Returns (dropna inside rolling handles NaNs)
    data['Returns'] = data['Close'].pct_change()
    
    # 1. Realized Volatility (21-day annualized)
    data['RV'] = data['Returns'].rolling(window_short).std() * np.sqrt(252)
    
    # 2. Skewness (negative = crash risk)
    data['Skew'] = data['Returns'].rolling(window_short).skew()
    
    # 3. Kurtosis (high = fat tails)
    data['Kurt'] = data['Returns'].rolling(window_short).kurt()
    
    # 4. Volume Proxy (short vs long-term vol ratio)
    data['Vol_Proxy'] = (
        data['Returns'].rolling(window_short).std() / 
        data['Returns'].rolling(window_long).std()
    )
    
    # Keep only our features + Returns + Date/Ticker
    data = data[['Returns', 'RV', 'Skew', 'Kurt', 'Vol_Proxy']].dropna()
    data['Ticker'] = ticker
    features_list.append(data.reset_index().set_index(['Ticker', 'Date']))

features = pd.concat(features_list)
print(f"\nFeatures shape: {features.shape}")
print(features.head())
print("\nNulls:\n", features.isnull().sum())
print("\nSPY sample:\n", features.xs('SPY', level=0).head())

# Export to features.csv
features.to_csv("features.csv")