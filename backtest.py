"""
BACKTEST - 4 PORTFOLIO COMPARISON
✅ Buy & Hold | ✅ Past Vol | ✅ GARCH | ✅ GAT
Daily long-short strategies | Exact test period | Production metrics
"""

import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== 4-PORTFOLIO BACKTEST FRAMEWORK ===")

# === 1. LOAD DATA & ALIGN WITH MODEL.PY ===
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([
    features.index.levels[0], 
    pd.to_datetime(features.index.levels[1])
])
features = features.sort_index()

# Get tradeable tickers
tradeable_tickers = [
    # ETFs (40)
    "SPY", "QQQ", "IWM", "MDY", "TLT", "IEF", "SHY", "TIP",
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLU", "XLB", "XLC", "XLRE",
    "USO", "GLD", "SLV", "DBC", "EWJ", "EWG", "EWU", "FXI", "EEM", "EFA", "ACWX",
    
    # Stocks (11 mega-caps)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "XOM", "UNH", "LLY"
]
print(f"Tradeable tickers: {len(tradeable_tickers)}")

# Load GAT results for exact test period
with open('gat_1month_test_results.pkl', 'rb') as f:
    gat_results = pickle.load(f)

test_dates = pd.DatetimeIndex(gat_results['dates'])
print(f"Test period: {test_dates[0].date()} → {test_dates[-1].date()} ({len(test_dates)} days)")

# === 2. EXTRACT DAILY RETURNS (Forward returns - no lookahead) ===
returns_data = features['Returns'].unstack('Ticker')
returns_data = returns_data.reindex(test_dates).shift(-1)  # Next day returns

# === 3. GARCH ROLLING FORECASTS (with fallback) ===
print("\n=== Computing GARCH Predictions ===")
rv_data = features['RV'].unstack('Ticker')
portfolio_returns = features['Returns'].groupby('Date').mean()

def garch_forecast_with_fallback(target_date, returns_series, fallback_rv):
    """GARCH forecast with past vol fallback"""
    train_returns = returns_series[returns_series.index < target_date].dropna() * 100
    
    if len(train_returns) < 100:
        return fallback_rv
    
    try:
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        forecast_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1]['h.1']) / 100
        return forecast_vol.item()
    except:
        return fallback_rv

# Pre-compute GARCH for portfolio
garch_portfolio_preds = {}
for date in test_dates:
    fallback = rv_data.loc[:date].mean(axis=1).iloc[-1] if date in rv_data.index else 0.15
    pred = garch_forecast_with_fallback(date, portfolio_returns, fallback)
    garch_portfolio_preds[date] = pred

print(f"✓ GARCH forecasts: {len(garch_portfolio_preds)} days")

# === 4. GAT PREDICTIONS (pre-loaded) ===
gat_preds_dict = dict(zip(test_dates, gat_results['predicted_rv']))
print(f"✓ GAT predictions: {len(gat_preds_dict)} days")

# === 5. DAILY SIGNALS GENERATION ===
def get_daily_signals(date, signal_type='past_vol'):
    """
    Generate signals for all tickers on given date (NO LOOKAHEAD)
    Returns: Series with ticker signals (lower = safer = long)
    """
    if date not in features.index.get_level_values('Date'):
        return pd.Series(dtype=float)
    
    # Get data UP TO date (previous close)
    past_data = features[features.index.get_level_values('Date') <= date]
    
    if signal_type == 'past_vol':
        # Today's realized vol
        signals = past_data.groupby('Ticker')['RV'].last()
    
    elif signal_type == 'garch':
        # Per-ticker GARCH (expensive, so use portfolio GARCH + relative RV)
        portfolio_garch = garch_portfolio_preds.get(date, 0.15)
        current_rv = past_data.groupby('Ticker')['RV'].last()
        # Scale by relative vol to portfolio
        signals = current_rv * (portfolio_garch / current_rv.mean())
    
    elif signal_type == 'gat':
        # GAT portfolio pred scaled by ticker RV
        gat_pred = gat_preds_dict.get(date, 0.15)
        current_rv = past_data.groupby('Ticker')['RV'].last()
        signals = current_rv * (gat_pred / current_rv.mean())
    
    return signals.dropna()

# === 6. PORTFOLIO CONSTRUCTION ===
def construct_long_short_portfolio(signals, available_tickers):
    """
    Long bottom 50% (low vol), Short top 50% (high vol)
    Equal weight within long/short legs
    """
    if len(signals) < 2:
        return pd.Series(dtype=float)
    
    # Filter to available tickers
    signals = signals[signals.index.isin(available_tickers)]
    
    # Rank and split
    n_half = len(signals) // 2
    sorted_tickers = signals.sort_values().index
    
    long_tickers = sorted_tickers[:n_half]
    short_tickers = sorted_tickers[-n_half:]
    
    # Equal weight
    weights = pd.Series(0.0, index=signals.index)
    weights.loc[long_tickers] = 1.0 / len(long_tickers)
    weights.loc[short_tickers] = -1.0 / len(short_tickers)
    
    return weights

# === 7. DAILY BACKTESTING ENGINE ===
print("\n=== Running Backtest ===")

portfolios = {
    'buy_hold': [],
    'past_vol': [],
    'garch': [],
    'gat': []
}

weights_history = {k: [] for k in portfolios.keys()}
tx_costs = 0.001  # 10bps per trade

for i, date in enumerate(test_dates[:-1]):  # Exclude last (no forward return)
    
    # Get available tickers and returns for NEXT day
    next_date = test_dates[i + 1]
    if next_date not in returns_data.index:
        continue
    
    daily_returns = returns_data.loc[next_date].dropna()
    available_tickers = daily_returns.index.tolist()
    
    if len(available_tickers) < 10:  # Skip if too few tickers
        continue
    
    # === PORTFOLIO 1: BUY & HOLD ===
    buy_hold_weights = pd.Series(1.0 / len(available_tickers), index=available_tickers)
    buy_hold_ret = (buy_hold_weights * daily_returns).sum()
    portfolios['buy_hold'].append(buy_hold_ret)
    weights_history['buy_hold'].append(buy_hold_weights)
    
    # === PORTFOLIO 2: PAST VOL ===
    past_signals = get_daily_signals(date, 'past_vol')
    past_weights = construct_long_short_portfolio(past_signals, available_tickers)
    
    if not past_weights.empty:
        past_ret = (past_weights * daily_returns).sum()
        # Transaction costs (turnover)
        if i > 0:
            prev_weights = weights_history['past_vol'][-1].reindex(past_weights.index, fill_value=0)
            turnover = (past_weights - prev_weights).abs().sum()
            past_ret -= turnover * tx_costs
        portfolios['past_vol'].append(past_ret)
        weights_history['past_vol'].append(past_weights)
    else:
        portfolios['past_vol'].append(0.0)
        weights_history['past_vol'].append(pd.Series(dtype=float))
    
    # === PORTFOLIO 3: GARCH ===
    garch_signals = get_daily_signals(date, 'garch')
    garch_weights = construct_long_short_portfolio(garch_signals, available_tickers)
    
    if not garch_weights.empty:
        garch_ret = (garch_weights * daily_returns).sum()
        if i > 0:
            prev_weights = weights_history['garch'][-1].reindex(garch_weights.index, fill_value=0)
            turnover = (garch_weights - prev_weights).abs().sum()
            garch_ret -= turnover * tx_costs
        portfolios['garch'].append(garch_ret)
        weights_history['garch'].append(garch_weights)
    else:
        portfolios['garch'].append(0.0)
        weights_history['garch'].append(pd.Series(dtype=float))
    
    # === PORTFOLIO 4: GAT ===
    gat_signals = get_daily_signals(date, 'gat')
    gat_weights = construct_long_short_portfolio(gat_signals, available_tickers)
    
    if not gat_weights.empty:
        gat_ret = (gat_weights * daily_returns).sum()
        if i > 0:
            prev_weights = weights_history['gat'][-1].reindex(gat_weights.index, fill_value=0)
            turnover = (gat_weights - prev_weights).abs().sum()
            gat_ret -= turnover * tx_costs
        portfolios['gat'].append(gat_ret)
        weights_history['gat'].append(gat_weights)
    else:
        portfolios['gat'].append(0.0)
        weights_history['gat'].append(pd.Series(dtype=float))

print(f"✓ Backtest complete: {len(portfolios['buy_hold'])} days")

# === 8. PERFORMANCE METRICS ===
def calculate_metrics(returns_series):
    """Comprehensive performance metrics"""
    returns = np.array(returns_series)
    
    # Cumulative
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns[-1] - 1
    
    # Annualized
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = returns.std() * np.sqrt(252)
    
    # Risk-adjusted
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    sortino = ann_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
    
    # Drawdown
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # Win stats
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    
    return {
        'Total Return': total_return,
        'Ann. Return': ann_return,
        'Ann. Vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD': max_dd,
        'Calmar': calmar,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss
    }

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

metrics_df = pd.DataFrame()
for name, rets in portfolios.items():
    metrics = calculate_metrics(rets)
    metrics_df[name.upper().replace('_', ' ')] = pd.Series(metrics)

print(metrics_df.to_string(float_format=lambda x: f'{x:.4f}'))

# === 9. VISUALIZATION ===
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Equity curves
axes[0, 0].plot((1 + np.array(portfolios['buy_hold'])).cumprod(), label='Buy & Hold', linewidth=2)
axes[0, 0].plot((1 + np.array(portfolios['past_vol'])).cumprod(), label='Past Vol', linewidth=2)
axes[0, 0].plot((1 + np.array(portfolios['garch'])).cumprod(), label='GARCH', linewidth=2)
axes[0, 0].plot((1 + np.array(portfolios['gat'])).cumprod(), label='GAT', linewidth=3)
axes[0, 0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Sharpe comparison
sharpes = [metrics_df.loc['Sharpe', col] for col in metrics_df.columns]
axes[0, 1].bar(metrics_df.columns, sharpes, color=['gray', 'blue', 'orange', 'green'])
axes[0, 1].set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
axes[0, 1].axhline(0, color='r', linestyle='--')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Drawdown comparison
for name in portfolios.keys():
    cum = (1 + np.array(portfolios[name])).cumprod()
    cum_max = np.maximum.accumulate(cum)
    dd = (cum - cum_max) / cum_max
    axes[1, 0].plot(dd, label=name.replace('_', ' ').title(), alpha=0.8)
axes[1, 0].set_title('Drawdowns', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Return distribution
axes[1, 1].hist(portfolios['buy_hold'], bins=30, alpha=0.5, label='Buy Hold', density=True)
axes[1, 1].hist(portfolios['past_vol'], bins=30, alpha=0.5, label='Past Vol', density=True)
axes[1, 1].hist(portfolios['garch'], bins=30, alpha=0.5, label='GARCH', density=True)
axes[1, 1].hist(portfolios['gat'], bins=30, alpha=0.5, label='GAT', density=True)
axes[1, 1].set_title('Return Distributions', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('backtest_4portfolios.png', dpi=300, bbox_inches='tight')
plt.show()

# === 10. SAVE RESULTS ===
backtest_results = {
    'portfolios': portfolios,
    'metrics': metrics_df.to_dict(),
    'test_dates': test_dates.tolist(),
    'tx_costs': tx_costs
}

with open('backtest_results.pkl', 'wb') as f:
    pickle.dump(backtest_results, f)

print("\n🎉 BACKTEST COMPLETE!")
print("Files generated:")
print("  backtest_4portfolios.png (performance dashboard)")
print("  backtest_results.pkl (full results)")
print("\n✅ Production ready!")
