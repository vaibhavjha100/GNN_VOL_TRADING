"""
GARCH(1,1) Benchmark - PERFECTLY ALIGNED WITH model.py
✅ Exact same splits/dates/targets/scaler | ✅ Rolling OOS | ✅ Production ready
"""

import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== GARCH(1,1) vs GAT - EXACT MATCH ===")

# === 1. LOAD IDENTICAL DATA ===
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([features.index.levels[0], pd.to_datetime(features.index.levels[1])])
features = features.sort_index()

# EXACT SAME TARGET CREATION
rv_wide = features['RV'].unstack('Ticker').dropna()
targets = rv_wide.shift(-1).mean(axis=1).dropna()  # Next-day portfolio mean RV
print(f"Targets: {len(targets)} days (μ={targets.mean():.4f}, σ={targets.std():.4f})")

# === 2. EXACT SAME SPLITS ===
train_end = pd.Timestamp('2024-01-01')
val_end = pd.Timestamp('2024-12-01')      # Matches your model.py
test_start = pd.Timestamp('2024-12-01')

train_mask = targets.index < train_end
val_mask = (targets.index >= train_end) & (targets.index < val_end)
test_mask = targets.index >= test_start

train_targets, val_targets, test_targets = targets[train_mask], targets[val_mask], targets[test_mask]
print(f"Splits: Train {len(train_targets)}d | Val {len(val_targets)}d | Test {len(test_targets)}d ✓")

# === 3. SAME GAT RESULTS ===
print("\n=== Loading GAT Results ===")
with open('gat_1month_test_results.pkl', 'rb') as f:
    gat_results = pickle.load(f)

test_dates = np.array(gat_results['dates'])
test_actual_scaled = np.array(gat_results['actual_scaled'])
test_gat_scaled = np.array(gat_results['predicted_scaled'])
test_actual_raw = np.array(gat_results['actual_rv'])
test_gat_raw = np.array(gat_results['predicted_rv'])

print(f"GAT Test: RMSE={np.sqrt(mean_squared_error(test_actual_scaled, test_gat_scaled)):.4f}, Corr={np.corrcoef(test_gat_scaled, test_actual_scaled)[0,1]:.4f}")

# === 4. GARCH(1,1) - IDENTICAL TEST PERIOD ===
print("\n=== GARCH Rolling Forecasts (Same Test Period) ===")

# Portfolio returns (daily) for GARCH input
daily_returns = features['Returns'].groupby('Date').mean()
daily_returns = daily_returns.reindex(targets.index).fillna(0)

def garch_forecast(target_date, returns_series):
    """Single forecast for exact GAT test date"""
    train_returns = returns_series[returns_series.index < target_date].dropna() * 100  # ARCH % scale
    
    if len(train_returns) < 100:
        return np.nan
    
    try:
        model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        # 1-step ahead volatility forecast (unscaled to match RV)
        forecast_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1]['h.1']) / 100
        return forecast_vol.item()
    except:
        return np.nan

# Generate GARCH predictions for EXACT GAT test dates
garch_preds_raw = []
for date in test_dates:
    pred = garch_forecast(pd.Timestamp(date), daily_returns)
    garch_preds_raw.append(pred)

garch_preds_raw = np.array(garch_preds_raw)

# === 5. SAME SCALER AS GAT ===
scaler = StandardScaler()
scaler.fit(train_targets.values.reshape(-1, 1))  # Fit on train only!
garch_preds_scaled = scaler.transform(garch_preds_raw.reshape(-1, 1)).flatten()

# Align (remove any NaNs)
valid_mask = ~np.isnan(garch_preds_scaled)
n_valid = np.sum(valid_mask)

garch_valid_scaled = garch_preds_scaled[valid_mask]
garch_valid_raw = garch_preds_raw[valid_mask]
gat_valid_scaled = test_gat_scaled[valid_mask]
actual_valid_scaled = test_actual_scaled[valid_mask]
actual_valid_raw = test_actual_raw[valid_mask]
gat_valid_raw = test_gat_raw[valid_mask]

print(f"Valid comparisons: {n_valid}/{len(test_dates)} dates")

# === 6. HEAD-TO-HEAD METRICS ===
garch_rmse_scaled = np.sqrt(mean_squared_error(actual_valid_scaled, garch_valid_scaled))
gat_rmse_scaled = np.sqrt(mean_squared_error(actual_valid_scaled, gat_valid_scaled))

garch_rmse_raw = np.sqrt(mean_squared_error(actual_valid_raw, garch_valid_raw))
gat_rmse_raw = np.sqrt(mean_squared_error(actual_valid_raw, gat_valid_raw))

garch_corr = np.corrcoef(garch_valid_scaled, actual_valid_scaled)[0, 1]
gat_corr = np.corrcoef(gat_valid_scaled, actual_valid_scaled)[0, 1]

print("\n" + "="*60)
print("FINAL BENCHMARK RESULTS")
print("="*60)
print(f"{'Metric':<20} {'GARCH':<10} {'GAT':<10} {'Winner'}")
print("-"*60)
print(f"{'Test RMSE (scaled)':<20} {garch_rmse_scaled:<10.4f} {gat_rmse_scaled:<10.4f} {'🟢 GAT' if gat_rmse_scaled < garch_rmse_scaled else '🔴 GARCH'}")
print(f"{'Test RMSE (raw)':<20} {garch_rmse_raw:<10.4f} {gat_rmse_raw:<10.4f} {'🟢 GAT' if gat_rmse_raw < garch_rmse_raw else '🔴 GARCH'}")
#print(f"{'Test Corr':<20} {garch_corr:<10.4f} {gat_corr:<10.4f} {'🟢 GAT' if abs(gat_corr) > abs(garch_corr) else '🔴 GARCH'}")
print(f"{'N samples':<20} {n_valid:<10} {n_valid:<10} {'TIE'}")
print("="*60)

# === 7. PRODUCTION VISUALIZATION ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Time series (exact test period)
valid_dates = [test_dates[i] for i in range(len(valid_mask)) if valid_mask[i]]
axes[0, 0].plot(valid_dates, actual_valid_raw, 'k-', label='Actual RV', linewidth=2)
axes[0, 0].plot(valid_dates, garch_valid_raw, 'b--', label='GARCH', alpha=0.8)
axes[0, 0].plot(valid_dates, gat_valid_raw, 'g:', label='GAT', linewidth=3)
axes[0, 0].set_title('OOS Test Period - Exact Match', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# 2-3. Scatter plots
minv, maxv = -2, 2
axes[0, 1].scatter(actual_valid_scaled, garch_valid_scaled, alpha=0.7, s=40, c='blue', edgecolors='white')
axes[0, 1].plot([minv, maxv], [minv, maxv], 'r--', lw=2)
axes[0, 1].set_title(f'GARCH vs Actual\nRMSE: {garch_rmse_scaled:.3f}')
axes[0, 1].set_xlabel('Actual (scaled)'); axes[0, 1].set_ylabel('GARCH Pred')

axes[1, 0].scatter(actual_valid_scaled, gat_valid_scaled, alpha=0.7, s=40, c='green', edgecolors='white')
axes[1, 0].plot([minv, maxv], [minv, maxv], 'r--', lw=2)
axes[1, 0].set_title(f'GAT vs Actual\nRMSE: {gat_rmse_scaled:.3f}')
axes[1, 0].set_xlabel('Actual (scaled)'); axes[1, 0].set_ylabel('GAT Pred')

# 4. Residuals comparison
axes[1, 1].scatter(actual_valid_scaled, actual_valid_scaled - garch_valid_scaled, alpha=0.7, s=30, label='GARCH')
axes[1, 1].scatter(actual_valid_scaled, actual_valid_scaled - gat_valid_scaled, alpha=0.7, s=30, label='GAT')
axes[1, 1].axhline(0, color='r', lw=2, label='Perfect')
axes[1, 1].set_title('Residuals (Pred - Actual)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('garch_vs_gat_exact.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 PERFECT BENCHMARK COMPLETE!")
print("✅ Uses EXACT same test dates/scaler/targets as model.py")
print("✅ Saved: garch_vs_gat_exact.png")