import pandas as pd

df = pd.read_csv("data.csv", index_col=[0,1])
df.index = df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])

# Sort the date index
df = df.sort_index(level=1)

# Slice for ^VIX ticker
df_vix = df.xs("^VIX", level=0)

print(df_vix.head())
print(df_vix.info())

# Check for null values

print(df_vix.isnull().sum())

# Get unique tickers from the MultiIndex (level 0)
unique_tickers = df.index.get_level_values(0).unique()
print(unique_tickers)
print(len(unique_tickers))

'''for ticker in unique_tickers:
    data = df.xs(ticker, level=0)
    print(data.isnull().sum())'''
