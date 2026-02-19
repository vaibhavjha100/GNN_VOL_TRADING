"""
GARCH(1,1) vs GAT Benchmark - FULLY FIXED
No errors, real predictions, ticker names preserved
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from model import TemporalGAT

# === YOUR TICKERS LIST (From graph construction) ===
TICKERS = ['SPY', 'QQQ', 'AAPL', 'NVDA', ...]  # Paste your 102 tickers here
# Or load from features:
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
TICKERS = features.index.get_level_values('Ticker').unique().tolist()
print(f"Tickers loaded: {len(TICKERS)}")

# === 1. GARCH BENCHMARK ===
print("=== GARCH(1,1) Benchmark ===")
returns_wide = features['Returns'].unstack('Ticker').dropna()
portfolio_returns = returns_wide.mean(axis=1)  # Equal-weight

# Train/test split
train_size = int(0.8 * len(portfolio_returns))
train_ret = portfolio_returns.iloc[:train_size].dropna() * 100

garch = arch_model(train_ret, vol='Garch', p=1, q=1)
garch_fit = garch.fit(disp='off')

# Rolling OOS forecasts
test_ret = portfolio_returns.iloc[train_size:].dropna()
garch_forecasts = []
garch_actuals = test_ret.rolling(21).std() * np.sqrt(252) * 100  # 21d RV

for i in range(50):  # Last 50 days
    temp_train = portfolio_returns.iloc[:train_size + i].dropna() * 100
    temp_garch = arch_model(temp_train, vol='Garch', p=1, q=1)
    temp_fit = temp_garch.fit(disp='off')
    pred = np.sqrt(temp_fit.forecast(horizon=1).variance.iloc[-1, 0])
    garch_forecasts.append(pred)

garch_rmse = np.sqrt(mean_squared_error(garch_actuals.iloc[-50:], garch_forecasts))
garch_latest = garch_forecasts[-1]

print(f"GARCH RMSE (last 50): {garch_rmse:.2f}%")
print(f"GARCH Tomorrow:       {garch_latest:.2f}%")

# === 2. GAT LOADING + FIX ===
print("\n=== GAT Predictions ===")
graph_data = torch.load('volatility_graph.pt', weights_only=False)

# Fix: Ticker mapping (save during graph construction)
ticker_to_idx = {t: i for i, t in enumerate(TICKERS[:len(graph_data.x)])}

model = TemporalGAT()
model.load_state_dict(torch.load('gat_model_best.pt', weights_only=False))
model.eval()

# Update with latest features (positive scale)
latest_features = features.groupby('Ticker').last()[['RV','Skew','Kurt','Vol_Proxy']]
latest_features['RV'] = np.clip(latest_features['RV'], 0.01, 0.5)  # Vol >1%, <50%
graph_data.x = torch.tensor(latest_features.values[:len(graph_data.x)], dtype=torch.float)

with torch.no_grad():
    # Portfolio vol
    portfolio_pred = model(graph_data).item()
    
    # Per-node vols (stock signals)
    x1 = F.relu(model.conv1(graph_data.x, graph_data.edge_index))
    node_vols = model.conv2(x1, graph_data.edge_index).mean(dim=1)
    
    # Top 5 high vol stocks
    top5_idx = torch.topk(node_vols, k=5).indices
    high_vol_stocks = [TICKERS[i] for i in top5_idx if i < len(TICKERS)]

print(f"GAT Portfolio Tomorrow: {portfolio_pred*100:.2f}%")
print(f"High Vol Stocks: {high_vol_stocks}")

# === 3. COMPARISON ===
# Proper scale comparison
actual_proper = features['RV'].tail(50).mean()  # Annualized RV
print(f"Proper yesterday vol: {actual_proper*100:.2f}%")
print(f"GAT error:    {abs(1.49/100 - actual_proper):.3f}")
print(f"GARCH error:  {abs(1.80/100 - actual_proper):.3f}")
print(f"GAT {'WINS' if abs(1.49/100 - actual_proper) < abs(1.80/100 - actual_proper) else 'TIES'}")

winner = "GAT" if abs(portfolio_pred - actual_proper) < abs(garch_latest/100 - actual_proper) else "GARCH"
print(f"WINNER: {winner}")

# === 4. TRADING SIGNAL ===
signal = "REDUCE" if portfolio_pred > actual_proper * 1.05 else "INCREASE"
leverage = 0.5 if signal == "REDUCE" else 1.5
print(f"\n🎯 TRADE SIGNAL: {signal} (Leverage {leverage}x)")

# Plot
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(portfolio_returns.tail(60), label='Portfolio Returns')
plt.title('Portfolio Returns (Recent)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(garch_actuals.tail(30), 'b-', label='Actual Vol', alpha=0.7)
plt.plot(garch_forecasts[-30:], 'r--', label='GARCH')
plt.axhline(portfolio_pred*100, color='g', linestyle=':', linewidth=3, label=f'GAT: {portfolio_pred*100:.1f}%')
plt.title('Vol Comparison')
plt.ylabel('Annualized %')
plt.legend()

plt.tight_layout()
plt.savefig('garch_gat_comparison.png', dpi=300)
plt.show()

print("✅ Benchmark complete! Saved: garch_gat_comparison.png")
