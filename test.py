import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# Load data (unchanged)
features = pd.read_csv("features.csv", index_col=[0,1])
features.index = features.index.set_levels([features.index.levels[0], pd.to_datetime(features.index.levels[1])])
features = features.sort_index()

tradeable_tickers = ["SPY", "QQQ", "IWM", "MDY", "TLT", "IEF", "SHY", "TIP", "XLY", "XLP", "XLE", "XLF", 
                    "XLV", "XLI", "XLK", "XLU", "XLB", "XLC", "XLRE", "USO", "GLD", "SLV", "DBC", 
                    "EWJ", "EWG", "EWU", "FXI", "EEM", "EFA", "ACWX", "AAPL", "MSFT", "NVDA", "AMZN", 
                    "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH", "LLY"]

features = features.loc[features.index.get_level_values(0).isin(tradeable_tickers)]

data = pd.read_csv("data.csv", index_col=[0,1])
data.index = data.index.set_levels([data.index.levels[0], pd.to_datetime(data.index.levels[1])])
data = data.sort_index()
data = data.loc[data.index.get_level_values(0).isin(tradeable_tickers)]

# Backtest period (unchanged)
start_date = pd.to_datetime('2024-12-01')
dates = sorted(features.index.get_level_values(1).unique())
dates = [d for d in dates if d >= start_date]
dates = pd.DatetimeIndex(dates)
end_date = max(dates)
print(f"Backtest: {start_date.date()} to {end_date.date()} ({len(dates)} days)")

# FIXED RETURNS PIPELINE - Single pct_change() only
prices_bt = (data['Close']
            .loc[data.index.get_level_values(0).isin(tradeable_tickers)]
            .loc[data.index.get_level_values(1).isin(dates)]
            .unstack(level=0)  # Pivot to wide format
            .reindex(dates))   # Align dates

returns_bt = prices_bt.pct_change().dropna(how='all', axis=1).fillna(0)
print(f"Returns_bt shape: {returns_bt.shape}")
print("Returns_bt extremes:", f"{returns_bt.min().min():.3f}", f"to {returns_bt.max().max():.3f}")
print("Returns_bt sample:\n", returns_bt.head())
print("NaN count per column:\n", returns_bt.isna().sum())

# Future RV (unchanged)
future_rv_raw = features['RV'].groupby(level=1).shift(-21)
future_rv_bt = future_rv_raw.loc[future_rv_raw.index.get_level_values(1).isin(dates)].dropna()
print(f"Future RV valid obs: {len(future_rv_bt)}")

# GARCH function (unchanged)
def fit_garch(rets_series, refit_window=252):
    forecasts = np.full(len(rets_series), np.nan)
    for i in range(refit_window, len(rets_series)):
        train_rets = rets_series.iloc[:i].dropna()
        if len(train_rets) < refit_window: continue
        try:
            model = arch_model(train_rets * 100, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off', show_warning=False)
            forecast = res.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.iloc[-1]['h.1']) / 100 * np.sqrt(252)
            forecasts[i] = vol
        except: continue
    return pd.Series(forecasts, index=rets_series.index)

# TRUE OOS GARCH - Full historical training
print("Fitting TRUE OOS GARCH...")

# FULL returns from data.csv (pre + backtest)
full_returns_raw = data['Close'].pct_change()
full_returns = full_returns_raw.loc[full_returns_raw.index.get_level_values(0).isin(tradeable_tickers)]
full_returns_wide = full_returns.unstack(level=0).dropna(how='all', axis=1)

garch_bt = pd.DataFrame(index=pd.MultiIndex.from_product([tradeable_tickers, dates]), 
                       columns=['GARCH_Vol']).fillna(np.nan)

train_cutoff = dates[0] - pd.Timedelta(days=1)
print(f"Full data available to: {full_returns_wide.index.max().date()}")
print(f"Training cutoff: {train_cutoff.date()}")

for ticker in tradeable_tickers:
    if ticker in full_returns_wide.columns:
        ticker_full = full_returns_wide[ticker].dropna()
        if len(ticker_full) >= 252:
            print(f"OOS {ticker}: {len(ticker_full)} days available")
            try:
                # Single model on ALL pre-backtest data
                train_data = ticker_full.loc[:train_cutoff].dropna()
                if len(train_data) >= 252:
                    model = arch_model(train_data * 100, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    
                    # Static forecast for entire backtest (simplest OOS)
                    for date in dates:
                        garch_bt.loc[(ticker, date), 'GARCH_Vol'] = res.conditional_volatility.iloc[-1] / 100 * np.sqrt(252)
            except Exception as e:
                print(f"{ticker} failed: {e}")
                continue

garch_bt = garch_bt.dropna()
print(f"TRUE OOS GARCH valid obs: {len(garch_bt)}")


# FIXED get_vol_value - Better multiindex handling
def get_vol_value(vol_measure, ticker_date):
    """Get volatility value for (ticker, date) tuple"""
    if isinstance(vol_measure, pd.Series):
        return vol_measure.get(ticker_date, np.nan)
    elif isinstance(vol_measure, pd.DataFrame) and 'GARCH_Vol' in vol_measure.columns:
        return vol_measure['GARCH_Vol'].get(ticker_date, np.nan)
    return np.nan

# CALCULATE GARCH FORECAST ACCURACY
print("\n=== GARCH FORECAST ACCURACY ===")

# Align GARCH predictions with actual future RV (same tickers/dates)
common_idx = future_rv_bt.index.intersection(garch_bt.index)
aligned_garch = garch_bt.loc[common_idx, 'GARCH_Vol']
aligned_actual = future_rv_bt.loc[common_idx]

print(f"Common observations for MAE: {len(common_idx)}")
print(f"GARCH MAE: {np.mean(np.abs(aligned_garch - aligned_actual)):.4f}")
print(f"GARCH MAPE: {np.mean(np.abs((aligned_garch - aligned_actual)/aligned_actual))*100:.1f}%")
print(f"GARCH correlation: {aligned_garch.corr(aligned_actual):.3f}")
print(f"Future RV range: {aligned_actual.min():.4f} to {aligned_actual.max():.4f}")
print(f"GARCH pred range: {aligned_garch.min():.4f} to {aligned_garch.max():.4f}")



# FIXED backtest_portfolio - Robust NaN handling + debugging
def backtest_portfolio(vol_measure, returns_df):
    positions = pd.DataFrame(0.0, index=dates, columns=tradeable_tickers)
    valid_days = 0
    
    for date in dates:
        ticker_vols = {}
        for ticker in tradeable_tickers:
            vol = get_vol_value(vol_measure, (ticker, date))
            if not pd.isna(vol):
                ticker_vols[ticker] = vol
        
        if len(ticker_vols) >= 2:
            # Select lowest volatility half
            low_vol_tickers = sorted(ticker_vols, key=ticker_vols.get)[:len(ticker_vols)//2]
            weight = 1.0 / len(low_vol_tickers)
            for ticker in low_vol_tickers:
                positions.loc[date, ticker] = weight
            valid_days += 1
        else:
            print(f"WARNING: Only {len(ticker_vols)} valid vols on {date.date()}")
    
    print(f"Valid position days: {valid_days}/{len(dates)}")
    
    positions = positions.ffill().fillna(0)  # Forward fill + fill remaining NaNs
    port_rets = (positions.shift(1) * returns_df).sum(axis=1)
    
    # Clean portfolio returns
    port_rets_clean = port_rets.dropna()
    print(f"Portfolio returns - valid days: {len(port_rets_clean)}, mean: {port_rets_clean.mean():.6f}, std: {port_rets_clean.std():.6f}")
    print(f"Coverage: {(positions.abs() > 0).sum().sum()}")
    
    return port_rets_clean, positions

print("Running backtests...")
actual_rets, actual_pos = backtest_portfolio(future_rv_bt, returns_bt)
garch_rets, garch_pos = backtest_portfolio(garch_bt, returns_bt)

# Metrics (unchanged)
def calculate_metrics(returns):
    clean = returns.dropna()
    if len(clean) < 10: 
        return {k: np.nan for k in ['Total Return', 'Ann Return', 'Ann Vol', 'Sharpe', 'Max DD']}
    cumrets = (1 + clean).cumprod()
    ann_ret = (1 + clean.mean()) ** 252 - 1
    ann_vol = clean.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0
    return {
        'Total Return': cumrets.iloc[-1] - 1,
        'Ann Return': ann_ret,
        'Ann Vol': ann_vol,
        'Sharpe': sharpe,
        'Max DD': -(cumrets / cumrets.expanding().max() - 1).min()
    }

actual_metrics = calculate_metrics(actual_rets)
garch_metrics = calculate_metrics(garch_rets)

metrics_df = pd.DataFrame({k: [actual_metrics.get(k, np.nan), garch_metrics.get(k, np.nan)] 
                        for k in ['Total Return', 'Ann Return', 'Ann Vol', 'Sharpe', 'Max DD']}).T
metrics_df.columns = ['Actual Future Vol', 'GARCH Pred']
print("\n=== BACKTEST RESULTS ===")
print(metrics_df.round(4))

diff_sharpe = actual_metrics.get('Sharpe', 0) - garch_metrics.get('Sharpe', 0)
print(f"\nLow-Vol Anomaly: {'YES' if actual_metrics.get('Sharpe', 0) > 0 else 'NO'} (Sharpe={actual_metrics.get('Sharpe', 0):.3f})")
print(f"Prediction Premium: {'YES' if diff_sharpe > 0 else 'NO'} (ΔSharpe={diff_sharpe:.3f})")
