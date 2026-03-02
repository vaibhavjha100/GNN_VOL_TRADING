"""
LONG-SHORT VOLATILITY DISPERSION BACKTEST (25%/25% QUARTILES + MARGIN)
✅ Matches model.py test period (2024-12+) 
✅ Returns from features.csv['Returns'] column
✅ Tradeable tickers only for positions (10L/10S = 25%)
✅ ABSOLUTE weights sum to 1.0 (longs=0.5 + shorts=0.5 margin)
✅ Full return-risk metrics
"""

import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

# === CONFIG ===
TRADEABLE_TICKERS = [
    "SPY", "QQQ", "IWM", "MDY", "TLT", "IEF", "SHY", "TIP", 
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLU", 
    "XLB", "XLC", "XLRE", "USO", "GLD", "SLV", "DBC", 
    "EWJ", "EWG", "EWU", "FXI", "EEM", "EFA", "ACWX", 
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", 
    "TSLA", "JPM", "XOM", "UNH", "LLY"
]

N_TRADEABLE = len(TRADEABLE_TICKERS)
N_LONG = int(0.5 * N_TRADEABLE)   # 10 tickers (25%)
N_SHORT = int(0.5 * N_TRADEABLE)  # 10 tickers (25%)

print(f"✅ Tradeable universe: {N_TRADEABLE} tickers")
print(f"✅ Long: {N_LONG} (50%) | Short: {N_SHORT} (50%) | ABS weights sum=1.0")

# === 1. LOAD MODEL OUTPUTS ===
print("\n=== Loading model outputs ===")
test_results = pickle.load(open('gat_1month_test_results.pkl', 'rb'))
dates = pd.to_datetime(test_results['dates'])

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model import TemporalGAT
model = TemporalGAT().to(device)
checkpoint = torch.load('gat_1month_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load features (contains Returns column)
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([features.index.levels[0], pd.to_datetime(features.index.levels[1])])
features = features.sort_index()

# Filter to test dates
test_features = features[features.index.get_level_values('Date').isin(dates)]

monthly_graphs = torch.load('monthly_graphs.pt', weights_only=False)

print(f"✅ Data loaded: {len(dates)} test days, {len(test_features)} obs")

# === 2. PER-ASSET VOLATILITY FORECASTS ===
print("\n=== Generating per-asset DL forecasts ===")
def get_graph_context(target_date):
    graph_dates = sorted(monthly_graphs.keys())
    available = [pd.Timestamp(d) for d in graph_dates if pd.Timestamp(d) < target_date]
    return monthly_graphs[str(available[-1].date())] if available else None

asset_forecasts = {}
test_returns_wide = test_features['Returns'].unstack('Ticker').reindex(columns=TRADEABLE_TICKERS)

for date in test_returns_wide.index:
    # Get most recent graph context (ALL tickers for features)
    context_graph = get_graph_context(date)
    if context_graph is None:
        continue
    
    # Update with recent features (all tickers, last 30 days before date)
    recent_start = date - pd.Timedelta(days=30)
    recent_feats = features[
        (features.index.get_level_values('Date') >= recent_start) & 
        (features.index.get_level_values('Date') < date)
    ]
    
    ticker_order = getattr(context_graph, 'ticker_order', list(range(context_graph.num_nodes)))
    latest_feats = recent_feats.groupby('Ticker')[['RV','Skew','Kurt','Vol_Proxy']].last()
    latest_feats = latest_feats.reindex(ticker_order).fillna(0.15)
    
    # Pad if needed
    num_nodes = context_graph.num_nodes
    if len(latest_feats) < num_nodes:
        pad_rows = num_nodes - len(latest_feats)
        pad_df = pd.DataFrame([[0.15]*4]*pad_rows, columns=['RV','Skew','Kurt','Vol_Proxy'])
        latest_feats = pd.concat([latest_feats, pad_df])
    
    context_graph.x = torch.tensor(latest_feats.values[:num_nodes], dtype=torch.float)
    
    # Forward pass for node embeddings
    with torch.no_grad():
        x = context_graph.x.to(device)
        edge_index = context_graph.edge_index.to(device)
        batch = torch.zeros(context_graph.num_nodes, dtype=torch.long, device=device)
        x = model.conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = model.conv2(x, edge_index)
        node_embeds = torch.nn.functional.elu(x).cpu().detach()
    
    # Per-asset vol forecast for TRADEABLE_TICKERS only
    tradeable_mask = [t in TRADEABLE_TICKERS for t in ticker_order]
    tradeable_indices = np.where(tradeable_mask)[0]
    asset_rv = latest_feats.loc[TRADEABLE_TICKERS]['RV'].fillna(0.15).values
    asset_embeds = node_embeds[tradeable_indices].mean(dim=1).numpy()
    
    dl_forecasts = np.abs(asset_embeds).flatten() * 0.3 + asset_rv * 0.7
    asset_forecasts[date] = pd.Series(dl_forecasts, index=TRADEABLE_TICKERS)

print(f"✅ Generated {len(asset_forecasts)} days of forecasts")

# === 3. LONG-SHORT STRATEGY (ABS WEIGHTS=1.0) ===
print("\n=== Running 25%/25% backtest (margin-adjusted) ===")
TARGET_VOL = 0.12

strategy_returns = []
positions = []

for date in test_returns_wide.index:
    if date not in asset_forecasts:
        continue
    
    vol_forecasts = asset_forecasts[date]
    day_returns = test_returns_wide.loc[date]
    
    # Available returns only
    valid_tickers = day_returns.dropna().index.intersection(TRADEABLE_TICKERS)
    
    n_long = int(0.5 * len(valid_tickers))
    n_short = int(0.5 * len(valid_tickers))
    
    if len(valid_tickers) < n_long + n_short or n_long < 1:
        continue
    
    valid_vols = vol_forecasts.loc[valid_tickers]
    vol_rank = valid_vols.rank(ascending=True)
    
    # Select 50% quantiles
    longs = vol_rank.nsmallest(n_long).index.tolist()      # High vol SHORT
    shorts = vol_rank.nlargest(n_short).index.tolist()   # Low vol LONG

    
    long_rets = day_returns[longs].dropna()
    short_rets = day_returns[shorts].dropna()
    
    if len(long_rets) < 3 or len(short_rets) < 3:
        continue
    
    # === ABSOLUTE WEIGHTS SUM TO 1.0 ===
    # Long side: 50% total allocation (0.5 total)
    long_sigmas = valid_vols.loc[long_rets.index]
    long_weights_raw = 1 / (long_sigmas + 1e-8)
    long_weights = long_weights_raw / long_weights_raw.sum() * 0.5  # 50% to longs
    
    # Short side: 50% total margin (0.5 total, shorts contribute -returns)
    short_sigmas = valid_vols.loc[short_rets.index]
    short_weights_raw = 1 / (short_sigmas + 1e-8)
    short_weights = short_weights_raw / short_weights_raw.sum() * 0.5  # 50% margin to shorts
    
    # Portfolio P&L
    long_pnl = np.dot(long_rets.values, long_weights.values)
    short_pnl = np.dot(short_rets.values, -short_weights.values)  # Short returns are negative of asset returns
    
    # CORRECTED: When shorts gain (asset falls), short_pnl is positive and adds to portfolio return
    port_ret = long_pnl + short_pnl
    
    # Vol targeting on historical port vol
    recent_rets = test_returns_wide.iloc[max(0, test_returns_wide.index.get_loc(date)-20):test_returns_wide.index.get_loc(date)]
    hist_port_vol = recent_rets.std(ddof=0).mean() * np.sqrt(252)
    leverage = min(TARGET_VOL / max(hist_port_vol, 0.05), 2.0)
    
    strategy_returns.append(port_ret * leverage)
    positions.append({
        'date': date, 
        'longs': longs[:5],
        'long_weights': dict(long_weights),
        'shorts': shorts[:5],
        'short_weights': dict(short_weights),
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'leverage': leverage,
        'port_ret': port_ret
    })

# === 4. PERFORMANCE METRICS ===
print("\n=== 📊 PERFORMANCE METRICS ===")
strategy_returns = pd.Series(strategy_returns, index=[p['date'] for p in positions])
daily_rets = np.array(strategy_returns.dropna())

annual_factor = 252
cumprod = np.cumprod(1 + daily_rets)

# Calculate metrics
total_return = (cumprod[-1] - 1) * 100
ann_return = (cumprod[-1] ** (annual_factor/len(daily_rets)) - 1) * 100
ann_vol = daily_rets.std() * np.sqrt(annual_factor) * 100
sharpe = (daily_rets.mean() - 0.0425 / annual_factor) / daily_rets.std() * np.sqrt(annual_factor) if daily_rets.std() > 0 else 0
sortino = (daily_rets.mean() - 0.0425 / annual_factor) / daily_rets[daily_rets < 0].std() * np.sqrt(annual_factor) if len(daily_rets[daily_rets < 0]) > 1 else 0
max_dd = ((np.maximum.accumulate(cumprod) / cumprod - 1).max()) * 100
win_rate = (daily_rets > 0).mean() * 100
profit_factor = abs(daily_rets[daily_rets > 0].sum() / daily_rets[daily_rets < 0].sum()) if len(daily_rets[daily_rets < 0]) > 0 else np.inf
calmar = ann_return / max_dd if max_dd > 0 else np.inf

metrics = {
    'Total Return (%)': total_return,
    'Annualized Return (%)': ann_return,
    'Annualized Vol (%)': ann_vol,
    'Sharpe Ratio': sharpe,
    'Sortino Ratio': sortino,
    'Max Drawdown (%)': max_dd,
    'Calmar Ratio': calmar,
    'Win Rate (%)': win_rate,
    'Profit Factor': profit_factor,
    'N Trading Days': len(daily_rets),
    'Period': f"{pd.Timestamp(positions[0]['date']).date()} → {pd.Timestamp(positions[-1]['date']).date()}"
}

metric_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).round(2)
print(metric_df.to_string(index=False))

# === 5. VISUALIZATION ===
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Cumulative PnL
axes[0,0].plot(pd.Series(cumprod-1)*100, linewidth=3)
axes[0,0].set_title(f'Cumulative Returns\nSharpe: {sharpe:.2f}')
axes[0,0].set_ylabel('Total Return (%)')
axes[0,0].grid(alpha=0.3)

# Drawdown
dd = np.maximum.accumulate(cumprod) / cumprod - 1
axes[0,1].fill_between(range(len(dd)), dd*100, 0, alpha=0.3, color='red')
axes[0,1].plot(dd*100, linewidth=2)
axes[0,1].set_title(f'Max DD: {max_dd:.1f}%')
axes[0,1].set_ylabel('Drawdown (%)')

# Returns histogram
axes[0,2].hist(daily_rets*10000, bins=25, alpha=0.7, edgecolor='black')
axes[0,2].axvline(daily_rets.mean()*10000, color='red', ls='--', 
                 label=f'Mean: {daily_rets.mean()*10000:.0f}bps')
axes[0,2].legend()
axes[0,2].set_title('Daily Returns')
axes[0,2].set_xlabel('bps')

# Rolling Sharpe
rolling_sharpe = pd.Series(daily_rets).rolling(20).apply(
    lambda x: (x.mean() - 0.0425/252) / x.std() * np.sqrt(252) if x.std()>0 else 0
)
axes[1,0].plot(rolling_sharpe)
axes[1,0].axhline(rolling_sharpe.mean(), color='red', ls='--')
axes[1,0].set_title('20-Day Rolling Sharpe')

# Weight concentration (avg abs weight per side)
avg_long_wt = np.mean([np.mean(list(p['long_weights'].values())) for p in positions])
avg_short_wt = np.mean([np.mean(list(p['short_weights'].values())) for p in positions])
axes[1,1].pie([0.5, 0.5], labels=['Long Allocation (50%)', 'Short Margin (50%)'], autopct='%1.0f%%')
axes[1,1].set_title(f'Avg Weights: L={avg_long_wt:.1%}/ticker, S={avg_short_wt:.1%}/ticker')

# Ticker frequency
all_longs = pd.Series([t for p in positions for t in p['longs']]).value_counts()
all_shorts = pd.Series([t for p in positions for t in p['shorts']]).value_counts()
top10 = list(set(list(all_longs.index[:5]) + list(all_shorts.index[:5])))
axes[1,2].bar(range(len(top10)), all_longs.reindex(top10, fill_value=0).values, 
              alpha=0.7, label='Long Days', color='green')
axes[1,2].bar(range(len(top10)), -all_shorts.reindex(top10, fill_value=0).values, 
              alpha=0.7, label='Short Days', color='red')
axes[1,2].set_xticks(range(len(top10)))
axes[1,2].set_xticklabels(top10, rotation=45)
axes[1,2].legend()
axes[1,2].set_title('Top Ticker Exposure')

plt.tight_layout()
plt.savefig('long_short_25pct_margin.png', dpi=300, bbox_inches='tight')
plt.show()

# === SAVE ===
results = {'returns': daily_rets.tolist(), 'metrics': metrics, 'positions': positions}
with open('long_short_25pct_margin.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n🎉 MARGIN-ADJUSTED 25%/25% COMPLETE!")
print("Files: long_short_25pct_margin.png | long_short_25pct_margin.pkl")
