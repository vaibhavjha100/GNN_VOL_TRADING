"""
Temporal GAT - DYNAMIC MONTHLY GRAPHS (1-MONTH SEQUENCE)
✅ Single Monthly Graph Input | ✅ Simplified Temporal | ✅ Production Ready
Uses monthly_graphs.pt → Context-aware vol forecasts
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

import pickle
import warnings
warnings.filterwarnings('ignore')

# === SIMPLIFIED 1-MONTH MODEL ===
class TemporalGAT(torch.nn.Module):
    def __init__(self, in_channels=4, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, dropout=dropout)
        # Simplified: No GRU (single month input)
        self.fc = torch.nn.Linear(hidden, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.l2_norm = 1.0
    
    def forward(self, graph_t):
        """
        graph_t: Single monthly graph → [batch_size] volatility predictions
        """
        x, edge_index, batch = graph_t.x, graph_t.edge_index, graph_t.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GAT layers with dropout
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout.p, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x_pool = global_mean_pool(x, batch)  # Portfolio embedding

        x_pool = F.normalize(x_pool, p=2, dim=-1) * self.l2_norm
        
        x_final = self.dropout(x_pool)
        return self.fc(x_final).squeeze(-1)  # [batch_size]

# === SIMPLIFIED COLLATION FOR SINGLE GRAPHS ===
def custom_collate(batch):
    """Simple collation: List of (graph, target) → (graphs, targets)"""
    graphs = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    return graphs, targets

if __name__ == "__main__":
    print("=== Temporal GAT - 1-MONTH SEQUENCES ===")

    # === 1. LOAD DATA ===
    monthly_graphs = torch.load('monthly_graphs.pt', weights_only=False)
    graph_dates = sorted(monthly_graphs.keys())
    print(f"✅ Loaded {len(monthly_graphs)} monthly graphs: {graph_dates[0]} → {graph_dates[-1]}")

    features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
    features.index = features.index.set_levels([
        features.index.levels[0], 
        pd.to_datetime(features.index.levels[1])
    ])
    features = features.sort_index()

    # Portfolio targets: next-day mean realized volatility
    rv_wide = features['RV'].unstack('Ticker').dropna()
    targets = rv_wide.shift(-1).mean(axis=1).dropna()
    print(f"Targets: {len(targets)} days (μ={targets.mean():.4f}, σ={targets.std():.4f})")

    # === 2. TEMPORAL SPLIT (No leakage) ===
    train_end = pd.Timestamp('2024-01-01')
    val_end = pd.Timestamp('2024-12-01')
    test_start = pd.Timestamp('2024-12-01')

    train_mask = targets.index < train_end
    val_mask = (targets.index >= train_end) & (targets.index < val_end)
    test_mask = targets.index >= test_start

    train_targets, val_targets, test_targets = (
        targets[train_mask], targets[val_mask], targets[test_mask]
    )
    print(f"Splits: Train {len(train_targets)}d | Val {len(val_targets)}d | Test {len(test_targets)}d ✓")

    # Save test targets
    with open('test_targets.pkl', 'wb') as f:
        pickle.dump({'dates': test_targets.index.tolist(), 'values': test_targets.values}, f)

    # Target scaling (fit on train only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_targets.values.reshape(-1,1)).flatten()
    val_scaled = scaler.transform(val_targets.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test_targets.values.reshape(-1,1)).flatten()

    # === 3. 1-MONTH GRAPH CONTEXT ===
    def get_graph_context(target_date):
        """Returns most recent monthly graph before target_date"""
        available_dates = [pd.Timestamp(d) for d in graph_dates if pd.Timestamp(d) < target_date]
        if len(available_dates) < 1:
            return None
        recent_date = str(available_dates[-1].date())
        return monthly_graphs[recent_date]

    # === 4. SINGLE-GRAPH DATA LOADER ===
    def create_sequence_loader(targets_scaled, mask, graph_context_func, features_df):
        data = []
        seq_len = 30
        
        valid_indices = np.where(mask)[0]
        for i, full_idx in enumerate(valid_indices):
            if i >= seq_len:  # Skip warmup period
                target_date = targets.index[full_idx]
                context_graph = graph_context_func(target_date)
                
                if context_graph is not None:
                    updated_graph = context_graph.clone()
                    
                    # Ticker alignment
                    ticker_order = getattr(context_graph, 'ticker_order', None)
                    if ticker_order is None:
                        ticker_order = list(range(context_graph.num_nodes))
                    
                    # Rolling 30-day features before target
                    recent_start = target_date - pd.Timedelta(days=seq_len)
                    recent_features = features_df[
                        (features_df.index.get_level_values('Date') >= recent_start) &
                        (features_df.index.get_level_values('Date') < target_date)
                    ]
                    
                    if len(recent_features) > 0:
                        latest_feats = recent_features.groupby('Ticker')[['RV','Skew','Kurt','Vol_Proxy']].last()
                        latest_feats = latest_feats.reindex(ticker_order).fillna(0.15)
                        
                        # Ensure exact shape match
                        num_nodes = context_graph.num_nodes
                        if len(latest_feats) < num_nodes:
                            pad_rows = num_nodes - len(latest_feats)
                            pad_df = pd.DataFrame([[0.15]*4]*pad_rows, columns=['RV','Skew','Kurt','Vol_Proxy'])
                            latest_feats = pd.concat([latest_feats, pad_df])
                        
                        updated_graph.x = torch.tensor(latest_feats.values[:num_nodes], dtype=torch.float)
                    
                    # (single graph, scaled target)
                    data.append((updated_graph, torch.tensor(float(targets_scaled[i]))))
        
        return DataLoader(data, batch_size=16, shuffle=True, collate_fn=custom_collate)  # Larger batch size OK now

    # Create all loaders
    train_loader = create_sequence_loader(train_scaled, train_mask, get_graph_context, features)
    val_loader = create_sequence_loader(val_scaled, val_mask, get_graph_context, features)
    test_loader = create_sequence_loader(test_scaled, test_mask, get_graph_context, features)
    
    print(f"✅ Loaders: Train {len(train_loader)} batches | Val {len(val_loader)} | Test {len(test_loader)}")

    # === 5. TRAINING WITH 1-MONTH SEQUENCES ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TemporalGAT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_rmse = float('inf')
    patience_counter = 0
    best_epoch = 0

    print("\n=== 🚀 Training 1-Month Temporal GAT ===")
    for epoch in range(300):
        # Training
        model.train()
        train_loss, train_count = 0, 0
        
        for batch_graphs, batch_targets in train_loader:
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            
            # 🔥 SIMPLIFIED: Direct batch processing (no per-sample loops)
            graph_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
            pred_batch = model(graph_batch)
            loss = criterion(pred_batch, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(pred_batch)
            train_count += len(pred_batch)
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_preds_total, val_actuals_total = [], []
        with torch.no_grad():
            for batch_graphs, batch_targets in val_loader:
                batch_targets_cpu = batch_targets.cpu()
                graph_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
                pred_batch = model(graph_batch).cpu()
                val_preds_total.extend(pred_batch.tolist())
                val_actuals_total.extend(batch_targets_cpu.numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_actuals_total, val_preds_total))
        val_losses.append(val_rmse)
        scheduler.step(val_rmse)

        # Logging
        if epoch % 25 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}: Train {avg_train_loss:.4f} | Val RMSE {val_rmse:.4f}")

        # Early stopping & checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'best_epoch': epoch,
                'best_rmse': val_rmse,
                'optimizer_state': optimizer.state_dict()
            }, 'gat_1month_best.pt')
            print(f"  🏆 New best Val RMSE: {val_rmse:.4f} (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter > 40:
                print(f"Early stopping at epoch {epoch} (patience exhausted)")
                break

    print(f"\n✅ Training complete! Best Val RMSE: {best_val_rmse:.4f} @ epoch {best_epoch}")

    # === 6. OOS TEST EVALUATION ===
    print("\n=== 🧪 OOS Test (2024H2-2026) ===")

    import torch.serialization
    torch.serialization.add_safe_globals([StandardScaler])

    checkpoint = torch.load('gat_1month_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds_scaled = []
    with torch.no_grad():
        for batch_graphs, _ in test_loader:
            graph_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
            pred_batch = model(graph_batch).cpu()
            test_preds_scaled.extend(pred_batch.tolist())
    
    # Comprehensive metrics
    test_len = len(test_preds_scaled)
    test_rmse_scaled = np.sqrt(mean_squared_error(test_scaled[:test_len], test_preds_scaled))
    
    test_actual_unscaled = scaler.inverse_transform(test_targets.values[:test_len].reshape(-1,1)).flatten()
    test_pred_unscaled = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1,1)).flatten()
    test_rmse_unscaled = np.sqrt(mean_squared_error(test_actual_unscaled, test_pred_unscaled))
    test_corr = np.corrcoef(test_preds_scaled, test_scaled[:test_len])[0,1]
    
    print(f"Test Results ({test_len} samples):")
    print(f"  RMSE (scaled):   {test_rmse_scaled:.4f}")
    print(f"  RMSE (unscaled): {test_rmse_unscaled:.4f}")
    print(f"  Correlation:      {test_corr:.4f}")
    print(f"  Val→Test gap:    {test_rmse_scaled/best_val_rmse:.1f}x")

    # Save results
    test_results = {
        'dates': test_targets.index[:test_len].tolist(),
        'actual_scaled': test_scaled[:test_len].tolist(),
        'predicted_scaled': test_preds_scaled,
        'actual_rv': test_actual_unscaled.tolist(),
        'predicted_rv': test_pred_unscaled.tolist(),
        'rmse_scaled': float(test_rmse_scaled),
        'rmse_unscaled': float(test_rmse_unscaled),
        'correlation': float(test_corr),
        'model_checkpoint': 'gat_1month_best.pt'
    }
    with open('gat_1month_test_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)

    # === 7. VISUALIZATION (Updated titles) ===
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves
    plt.subplot(2, 4, 1)
    plt.semilogy(train_losses[:150], label='Train Loss', alpha=0.8, linewidth=1)
    plt.plot(val_losses, label='Val RMSE', linewidth=3, marker='o', markersize=4)
    plt.axvline(best_epoch, color='green', linestyle='--', alpha=0.8, label=f'Best ({best_val_rmse:.3f})')
    plt.title('1-Month Temporal Training')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Val scatter
    plt.subplot(2, 4, 2)
    val_preds_array = np.array(val_preds_total)
    plt.scatter(val_actuals_total, val_preds_array, alpha=0.6, s=25, c='steelblue', edgecolors='white', linewidth=0.5)
    minv, maxv = min(np.min(val_actuals_total), np.min(val_preds_array)), max(np.max(val_actuals_total), np.max(val_preds_array))
    plt.plot([minv, maxv], [minv, maxv], 'r--', lw=2, alpha=0.8)
    plt.xlabel('Actual (scaled)')
    plt.ylabel('Predicted')
    plt.title(f'Validation\nRMSE: {best_val_rmse:.3f}')
    
    # 3. Test scatter (scaled)
    plt.subplot(2, 4, 3)
    plt.scatter(test_scaled[:test_len], test_preds_scaled, alpha=0.6, s=25, c='forestgreen', edgecolors='white', linewidth=0.5)
    plt.plot([minv, maxv], [minv, maxv], 'r--', lw=2, alpha=0.8)
    plt.xlabel('Actual (scaled)')
    plt.ylabel('Predicted')
    plt.title(f'Test OOS\nRMSE: {test_rmse_scaled:.3f}')
    
    # 4. Test unscaled
    plt.subplot(2, 4, 4)
    plt.scatter(test_actual_unscaled, test_pred_unscaled, alpha=0.6, s=25, c='darkorange', edgecolors='white', linewidth=0.5)
    plt.plot([0, 0.6], [0, 0.6], 'r--', lw=2, alpha=0.8)
    plt.xlabel('Actual RV')
    plt.ylabel('Pred RV')
    plt.title(f'Unscaled Test\nRMSE: {test_rmse_unscaled:.3f}')
    
    # 5. Residuals
    plt.subplot(2, 4, 5)
    residuals = np.array(test_preds_scaled) - test_scaled[:test_len]
    plt.scatter(test_scaled[:test_len], residuals, alpha=0.6, s=20, c='purple')
    plt.axhline(0, color='r', lw=2)
    plt.axhline(np.mean(residuals), color='orange', lw=1, label=f'Mean: {np.mean(residuals):.3f}')
    plt.xlabel('Actual (scaled)')
    plt.ylabel('Residuals')
    plt.title('Test Residuals')
    plt.legend()
    
    # 6. Prediction distribution
    plt.subplot(2, 4, 6)
    plt.hist(test_pred_unscaled, bins=25, alpha=0.7, label='Predicted', density=True, color='green')
    plt.hist(test_actual_unscaled, bins=25, alpha=0.7, label='Actual', density=True, color='orange')
    plt.xlabel('Realized Volatility')
    plt.ylabel('Density')
    plt.title('Pred vs Actual RV Dist.')
    plt.legend()
    
    # 7. Performance table
    plt.subplot(2, 4, 7)
    metrics_text = f'''1-Month GAT Results
Val RMSE: {best_val_rmse:.3f}
Test RMSE: {test_rmse_scaled:.3f}
Test Corr: {test_corr:.3f}
Gap: {test_rmse_scaled/best_val_rmse:.1f}x
N Test: {test_len}'''
    plt.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.title('Performance Summary')
    plt.axis('off')
    
    # 8. Temporal mechanism
    plt.subplot(2, 4, 8)
    plt.text(0.5, 0.5, '1-Month Context\n[Single Graph]\nGAT + Pooling ✓', ha='center', va='center', 
             fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.title('Temporal Mechanism')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gat_1month_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n🎉 1-MONTH TEMPORAL GAT COMPLETE!")
    print("Files generated:")
    print("  gat_1month_best.pt (full checkpoint)")
    print("  gat_1month_test_results.pkl (production ready)")
    print("  gat_1month_results.png (8-panel dashboard)")
    print("\n✅ Ready for backtest.py & production!")
