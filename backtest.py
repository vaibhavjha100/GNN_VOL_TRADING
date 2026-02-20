"""
1-YEAR FULL BACKTEST: GAT vs GARCH vs Buy-Hold
2025 Complete OOS + Prediction Metrics
Your data.csv + features.csv → Sharpe + RMSE + Win Rate
"""

import pandas as pd
import numpy as np
from arch import arch_model
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from model import TemporalGAT

warnings.filterwarnings('ignore')

print("=== 1-YEAR GAT vs GARCH BACKTEST (2025 Full) ===")

# === 1. LOAD DATA ===
print("Loading data...")
data = pd.read_csv('data.csv', index_col=['Ticker', 'Date'])
data.index = data.index.set_levels([data.index.levels[0], pd.to_datetime(data.index.levels[1])])
features = pd.read_csv('features.csv', index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([features.index.levels[0], pd.to_datetime(features.index.levels[1])])

# Portfolio returns (equal weight)
# Need to unstack to wide format for pct_change to work per ticker, or group by ticker
close_wide = data['Close'].unstack('Ticker')
returns_wide = close_wide.pct_change().dropna(how='all')
portfolio_returns = returns_wide.mean(axis=1)

# Get sorted list of tickers as used in graph construction
ALL_TICKERS = close_wide.columns.tolist()

# 2025 FULL test (252 trading days)
test_start = '2025-01-01'
test_end = '2025-12-31'
test_returns = portfolio_returns[test_start:test_end].dropna()
train_returns = portfolio_returns[:test_start].dropna()

print(f"Train: {len(train_returns)} days | Test 2025: {len(test_returns)} days")

if len(test_returns) == 0:
    print("WARNING: No test data found for 2025. Adjusting test period to latest available.")
    test_returns = portfolio_returns.tail(100)
    train_returns = portfolio_returns.iloc[:-100]

# Realized vol (21-day rolling)
# Using annualized vol
test_vol_actual = test_returns.rolling(21).std() * np.sqrt(252)

# === 2. GARCH ROLLING FORECASTS ===
print("\nComputing GARCH forecasts...")
garch_forecast_vols = []
garch_signals = []

for i in range(len(test_returns)):
    # Rolling retrain
    train_end_idx = len(train_returns) + i
    temp_train = portfolio_returns.iloc[:train_end_idx].dropna() * 100
    
    if len(temp_train) > 100:
        try:
            garch = arch_model(temp_train, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off')
            # forecast() gives results in % (since train * 100)
            pred_vol = np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0]) / 100
        except:
            pred_vol = 0.15
            
        # Signal logic
        # Handle NaN for early test days
        recent_vol_window = test_vol_actual.iloc[max(0, i-5):i+1].dropna()
        recent_vol = recent_vol_window.mean() if not recent_vol_window.empty else 0.15
        
        leverage = 1.5 if pred_vol < recent_vol * 1.05 else 0.5
        signal_ret = test_returns.iloc[i] * leverage
        
        garch_forecast_vols.append(pred_vol)
        garch_signals.append(signal_ret)
    else:
        garch_forecast_vols.append(0.15)
        garch_signals.append(test_returns.iloc[i])

# === 3. GAT ROLLING FORECASTS ===
print("Computing GAT forecasts...")
graph_data = torch.load('volatility_graph.pt', weights_only=False)
model = TemporalGAT()
model.load_state_dict(torch.load('gat_model_best.pt', weights_only=False))
model.eval()

gat_forecast_vols = []
gat_signals = []

for i, date in enumerate(test_returns.index):
    # Rolling features to this date
    temp_features = features[features.index.get_level_values('Date') <= date]
    
    if len(temp_features) > 252:
        # Latest node features (clipped)
        latest = temp_features.groupby('Ticker').last()[['RV','Skew','Kurt','Vol_Proxy']]
        # Reindex based on ALL_TICKERS to ensure graph alignment
        latest = latest.reindex(ALL_TICKERS[:len(graph_data.x)])
        latest = latest.ffill().fillna(0.15)
        latest['RV'] = np.clip(latest['RV'], 0.01, 0.5)
        
        graph_data.x = torch.tensor(latest.values[:len(graph_data.x)], dtype=torch.float)
        
        with torch.no_grad():
            pred_vol = model(graph_data).item()
        
        # Signal
        recent_vol_window = test_vol_actual.iloc[max(0, i-5):i+1].dropna()
        recent_vol = recent_vol_window.mean() if not recent_vol_window.empty else 0.15
        
        leverage = 1.5 if pred_vol < recent_vol * 1.05 else 0.5
        signal_ret = test_returns.iloc[i] * leverage
        
        gat_forecast_vols.append(pred_vol)
        gat_signals.append(signal_ret)
    else:
        gat_forecast_vols.append(0.15)
        gat_signals.append(test_returns.iloc[i])

# Truncate to match lengths
min_len = min(len(garch_signals), len(gat_signals), len(test_returns))
garch_signals = garch_signals[:min_len]
gat_signals = gat_signals[:min_len]
test_returns_bt = test_returns.iloc[:min_len]
test_vol_actual_bt = test_vol_actual.iloc[:min_len]
garch_forecast_vols = garch_forecast_vols[:min_len]
gat_forecast_vols = gat_forecast_vols[:min_len]

# === 4. PERFORMANCE METRICS ===
def calc_metrics(returns):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    cum_ret = (1 + returns).prod() - 1
    # Max Drawdown from returns series
    cum_pnl = (1 + returns).cumprod()
    max_dd = (cum_pnl.cummax() - cum_pnl).max() / cum_pnl.cummax().max() if not cum_pnl.empty else 0
    win_rate = (returns > 0).mean()
    return sharpe, cum_ret, max_dd, win_rate

bh_sharpe, bh_cum, bh_dd, bh_wr = calc_metrics(test_returns_bt)
garch_sharpe, garch_cum, garch_dd, garch_wr = calc_metrics(pd.Series(garch_signals))
gat_sharpe, gat_cum, gat_dd, gat_wr = calc_metrics(pd.Series(gat_signals))

# Vol prediction accuracy (Drop NaNs from rolling actual vol)
valid_mask = ~np.isnan(test_vol_actual_bt.values)
actual_valid = test_vol_actual_bt.values[valid_mask]
garch_pred_valid = np.array(garch_forecast_vols)[valid_mask]
gat_pred_valid = np.array(gat_forecast_vols)[valid_mask]

if len(actual_valid) > 0:
    garch_vol_rmse = np.sqrt(mean_squared_error(actual_valid, garch_pred_valid))
    garch_vol_mae = mean_absolute_error(actual_valid, garch_pred_valid)
    gat_vol_rmse = np.sqrt(mean_squared_error(actual_valid, gat_pred_valid))
    gat_vol_mae = mean_absolute_error(actual_valid, gat_pred_valid)
else:
    garch_vol_rmse = gat_vol_rmse = 0

# === 5. RESULTS TABLE ===
print("\n" + "="*80)
print("2025 FULL YEAR BACKTEST RESULTS")
print("="*80)
print(f"{'Strategy':<12} {'Sharpe':<8} {'CumRet':<8} {'MaxDD':<8} {'Win%':<6} {'VolRMSE':<8}")
print("-"*80)
print(f"{'Buy&Hold':<12} {bh_sharpe:<8.2f} {bh_cum:<8.1%} {bh_dd:<8.1%} {bh_wr:<6.1%} {'-':<8}")
print(f"{'GARCH':<12} {garch_sharpe:<8.2f} {garch_cum:<8.1%} {garch_dd:<8.1%} {garch_wr:<6.1%} {garch_vol_rmse:<8.3f}")
print(f"{'GAT':<12} {gat_sharpe:<8.2f} {gat_cum:<8.1%} {gat_dd:<8.1%} {gat_wr:<6.1%} {gat_vol_rmse:<8.3f}")
print("-"*80)

gat_vs_garch = "GAT" if gat_sharpe > garch_sharpe else "GARCH"
print(f"🏆 TRADING WINNER: {gat_vs_garch}")
vol_win = "GAT" if gat_vol_rmse < garch_vol_rmse else "GARCH"
print(f"🏆 VOL PRED WINNER: {vol_win}")

# === 6. PLOTS ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Cum returns
cum_gat = (1 + pd.Series(gat_signals)).cumprod() - 1
cum_garch = (1 + pd.Series(garch_signals)).cumprod() - 1
cum_bh = (1 + test_returns_bt.values).cumprod() - 1
axes[0,0].plot(cum_gat.values, label=f'GAT (Sharpe {gat_sharpe:.2f})', linewidth=2)
axes[0,0].plot(cum_garch.values, label=f'GARCH (Sharpe {garch_sharpe:.2f})')
axes[0,0].plot(cum_bh, label='Buy & Hold')
axes[0,0].set_title('Cumulative Returns (2025 Full Year)')
axes[0,0].legend()
axes[0,0].grid()

# Daily returns vs signals
axes[0,1].plot(range(len(test_returns_bt)), test_returns_bt.values, alpha=0.6, label='Returns')
axes[0,1].scatter(range(len(gat_signals)), gat_signals, c='green', s=20, label='GAT Signals', alpha=0.7)
axes[0,1].set_title('GAT Signals vs Actual Returns')
axes[0,1].legend()

# Rolling Sharpe
rolling_gat = pd.Series(gat_signals).rolling(21).apply(
    lambda x: x.mean()/x.std()*np.sqrt(252) if len(x)>0 and x.std()>0 else 0
)
rolling_garch = pd.Series(garch_signals).rolling(21).apply(
    lambda x: x.mean()/x.std()*np.sqrt(252) if len(x)>0 and x.std()>0 else 0
)
axes[1,0].plot(rolling_gat.values, label='GAT', linewidth=2)
axes[1,0].plot(rolling_garch.values, label='GARCH')
axes[1,0].axhline(gat_sharpe, color='green', linestyle='--')
axes[1,0].set_title('21-Day Rolling Sharpe')
axes[1,0].legend()

# Vol prediction accuracy
axes[1,1].scatter(garch_forecast_vols, test_vol_actual_bt.values, alpha=0.6, label='GARCH')
axes[1,1].scatter(gat_forecast_vols, test_vol_actual_bt.values, alpha=0.6, label='GAT')
axes[1,1].plot([0,0.6], [0,0.6], 'k--', label='Perfect')
axes[1,1].set_xlabel('Predicted Vol')
axes[1,1].set_ylabel('Actual Vol')
axes[1,1].set_title('Vol Prediction Accuracy')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('1year_full_backtest.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 1-YEAR BACKTEST COMPLETE!")
print("Saved: 1year_full_backtest.png")
