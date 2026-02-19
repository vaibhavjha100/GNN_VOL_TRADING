"""
COMPLETE GARCH vs GAT BACKTEST
Uses your attached files: data.csv + features.csv
2025 OOS test → Sharpe proof
"""

import pandas as pd
import numpy as np
from arch import arch_model
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from model import TemporalGAT

warnings.filterwarnings('ignore')

print("=== FULL GARCH vs GAT BACKTEST ===")

# === 1. LOAD YOUR FILES ===
print("Loading data...")
data = pd.read_csv('data.csv', index_col=['Ticker', 'Date'])
data.index = data.index.set_levels([data.index.levels[0], pd.to_datetime(data.index.levels[1])])
features = pd.read_csv('features.csv', index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([features.index.levels[0], pd.to_datetime(features.index.levels[1])])

print(f"Data shape: {data.shape}, Features: {features.shape}")

TICKERS = features.index.get_level_values('Ticker').unique().tolist()
print(f"Tickers: {len(TICKERS)}")

# === 2. PORTFOLIO RETURNS (Equal Weight) ===
# Unstack Close to wide format (dates × tickers), then compute returns
close_wide = data['Close'].unstack('Ticker')          # shape: dates × tickers
returns_wide = close_wide.pct_change().dropna(how='all')
portfolio_returns = returns_wide.mean(axis=1)          # Daily equal-weight portfolio P&L
portfolio_returns = portfolio_returns.sort_index()

# 2025 OOS test period
test_start = '2025-01-01'
test_returns = portfolio_returns[test_start:].dropna()
train_returns = portfolio_returns[:test_start].dropna()

print(f"Train: {len(train_returns)} days, Test: {len(test_returns)} days")

if len(test_returns) == 0:
    raise ValueError("No test data found after 2025-01-01. Check your data.csv date range.")

# === 3. GARCH(1,1) ROLLING FORECASTS ===
print("\nGARCH Rolling Forecasts...")
garch_signals = []
n_test = min(50, len(test_returns))

for i in range(n_test):
    train_end = len(train_returns) + i
    temp_train = portfolio_returns.iloc[:train_end].dropna() * 100  # % scale for arch

    if len(temp_train) > 100:
        garch = arch_model(temp_train, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        # pred_vol is in % units (matching temp_train scale)
        pred_vol_pct = np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0])

        # recent_vol also in % units for fair comparison
        window = temp_train.iloc[-21:]
        recent_vol_pct = window.std() if len(window) > 1 else 15.0

        leverage = 1.5 if pred_vol_pct < recent_vol_pct * 1.05 else 0.5
        signal_ret = test_returns.iloc[i] * leverage
        garch_signals.append(signal_ret)
    else:
        garch_signals.append(test_returns.iloc[i])  # Hold

garch_series = pd.Series(garch_signals)
garch_sharpe = garch_series.mean() / garch_series.std() * np.sqrt(252) if garch_series.std() > 0 else 0.0

# GARCH RMSE: use 21-day rolling RV of test returns as the "actual" vol
garch_actual_vols_raw = (test_returns * 100).rolling(21).std()
garch_actual_vols = garch_actual_vols_raw.iloc[:n_test].dropna()

garch_forecasts_trimmed = []
for i in range(n_test):
    train_end = len(train_returns) + i
    temp_train = portfolio_returns.iloc[:train_end].dropna() * 100
    if len(temp_train) > 100:
        garch = arch_model(temp_train, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        garch_forecasts_trimmed.append(np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0]))

# Align lengths for RMSE
min_len = min(len(garch_actual_vols), len(garch_forecasts_trimmed))
if min_len > 0:
    garch_rmse = np.sqrt(mean_squared_error(
        garch_actual_vols.values[-min_len:],
        garch_forecasts_trimmed[-min_len:]
    ))
    print(f"GARCH RMSE (vol, %): {garch_rmse:.4f}")

# === 4. GAT ROLLING PREDICTIONS ===
print("\nGAT Rolling Predictions...")
graph_data = torch.load('volatility_graph.pt', weights_only=False)
model = TemporalGAT()
model.load_state_dict(torch.load('gat_model_best.pt', weights_only=False))
model.eval()

gat_signals = []
test_dates = test_returns.index

for i, date in enumerate(test_dates[:n_test]):
    # Get features up to (and including) this date
    temp_features = features.loc[features.index.get_level_values('Date') <= date]

    if len(temp_features) > 252:
        # Latest node feature vector per ticker, aligned to graph node order
        latest_feat = (
            temp_features
            .groupby('Ticker')[['RV', 'Skew', 'Kurt', 'Vol_Proxy']]
            .last()
            .reindex(TICKERS[:graph_data.num_nodes])
            .fillna(0.15)
        )
        latest_feat['RV'] = np.clip(latest_feat['RV'], 0.01, 0.5)

        graph_data.x = torch.tensor(latest_feat.values, dtype=torch.float)

        with torch.no_grad():
            pred_vol = model(graph_data).item()  # Unitless (model output scale)

        # recent_vol in same scale as model output (raw RV, not %)
        recent_vol = temp_features['RV'].tail(21).mean()
        leverage = 1.5 if pred_vol < recent_vol * 1.05 else 0.5
        signal_ret = test_returns[date] * leverage
        gat_signals.append(signal_ret)
    else:
        gat_signals.append(test_returns[date])

gat_series = pd.Series(gat_signals)
gat_sharpe = gat_series.mean() / gat_series.std() * np.sqrt(252) if gat_series.std() > 0 else 0.0

# === 5. BUY & HOLD METRICS ===
bh_series = test_returns.iloc[:n_test]
bh_sharpe = bh_series.mean() / bh_series.std() * np.sqrt(252) if bh_series.std() > 0 else 0.0

# === 6. RESULTS ===
print("\n=== BACKTEST RESULTS (2025 OOS) ===")
print(f"{'='*55}")
print(f"Period: {test_returns.index[0].date()} → {test_returns.index[min(n_test-1, len(test_returns)-1)].date()} ({n_test} days)")
print(f"\n{'Strategy':<15} {'Sharpe':>7} {'Cum Return':>12}")
print(f"{'GAT':<15} {gat_sharpe:>7.2f} {np.prod(1 + gat_series) - 1:>11.1%}")
print(f"{'GARCH':<15} {garch_sharpe:>7.2f} {np.prod(1 + garch_series) - 1:>11.1%}")
print(f"{'Buy & Hold':<15} {bh_sharpe:>7.2f} {np.prod(1 + bh_series) - 1:>11.1%}")

winner = "GAT" if gat_sharpe > garch_sharpe else "GARCH"
print(f"\n🏆 {winner} WINS!")

# === 7. PLOT RESULTS ===
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
cum_gat   = np.cumprod(1 + gat_series) - 1
cum_garch = np.cumprod(1 + garch_series) - 1
cum_bh    = np.cumprod(1 + bh_series.values) - 1
plt.plot(cum_gat.values,   label=f'GAT (Sharpe {gat_sharpe:.2f})')
plt.plot(cum_garch.values, label=f'GARCH (Sharpe {garch_sharpe:.2f})')
plt.plot(cum_bh,           label=f'Buy & Hold (Sharpe {bh_sharpe:.2f})')
plt.title('Cumulative Returns (2025 OOS)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(bh_series.values, alpha=0.7, label='Daily Returns (B&H)')
plt.plot(gat_series.values, alpha=0.7, color='green', label='GAT Scaled')
plt.title('GAT Signals vs Returns')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
rolling_sharpe_gat = gat_series.rolling(min(21, n_test)).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
)
plt.plot(rolling_sharpe_gat.values)
plt.axhline(gat_sharpe, color='green', linestyle='--', label=f'Overall: {gat_sharpe:.2f}')
plt.title('GAT Rolling 21-Day Sharpe')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
errors = np.abs(gat_series.values - bh_series.values)
plt.hist(errors, bins=20, color='steelblue', edgecolor='white')
plt.title('GAT vs B&H Signal Errors')
plt.xlabel('|Error|')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('full_backtest_results.png', dpi=300)
plt.show()

print("\n✅ Full backtest complete!")
print("Files saved: full_backtest_results.png")
