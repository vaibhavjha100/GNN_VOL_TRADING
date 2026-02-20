"""
Temporal GAT - DYNAMIC MONTHLY GRAPHS + PROPER SPLIT
Uses monthly_graphs.pt → Context-aware vol forecasts
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# === ENHANCED TEMPORAL MODEL ===
class TemporalGAT(torch.nn.Module):
    def __init__(self, in_channels=4, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, dropout=dropout)
        # GRU for 3-month graph sequence
        self.temporal = torch.nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden, 1)
    
    def forward(self, graph_sequence):
        """
        graph_sequence: List[3 monthly graphs] with updated node features
        """
        embeddings = []
        for graph_t in graph_sequence:  # Process each monthly graph
            x, edge_index, batch = graph_t.x, graph_t.edge_index, graph_t.batch
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            # GAT layers
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x_pool = global_mean_pool(x, batch)  # Portfolio embedding
            embeddings.append(x_pool)
        
        # Temporal fusion (3 monthly contexts)
        # torch.stack(embeddings) → [seq, batch, hidden]; permute for batch_first GRU
        stacked = torch.stack(embeddings).permute(1, 0, 2)  # [batch, seq, hidden]
        x_gru, _ = self.temporal(stacked)
        return self.fc(x_gru[:, -1, :]).view(-1)  # Use final hidden state

if __name__ == "__main__":
    print("=== Temporal GAT - MONTHLY ROLLING GRAPHS ===")

    # === 1. LOAD DYNAMIC DATA ===
    monthly_graphs = torch.load('monthly_graphs.pt', weights_only=False)
    graph_dates = sorted(monthly_graphs.keys())
    print(f"Loaded {len(monthly_graphs)} monthly graphs: {graph_dates[0]} → {graph_dates[-1]}")

    features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
    features.index = features.index.set_levels([
        features.index.levels[0], 
        pd.to_datetime(features.index.levels[1])
    ])
    features = features.sort_index()

    # Portfolio targets
    rv_wide = features['RV'].unstack('Ticker').dropna()
    targets = rv_wide.shift(-1).mean(axis=1).dropna()
    print(f"Targets: {len(targets)} days")

    # === 2. TIME-ALIGNED SPLIT ===
    train_end = pd.Timestamp('2023-01-01')
    val_end = pd.Timestamp('2024-07-01')
    test_start = pd.Timestamp('2024-07-01')

    train_mask = targets.index < train_end
    val_mask = (targets.index >= train_end) & (targets.index < val_end)
    test_mask = targets.index >= test_start

    train_targets, val_targets, test_targets = (
        targets[train_mask], targets[val_mask], targets[test_mask]
    )

    print(f"Train: {len(train_targets)}d | Val: {len(val_targets)}d | Test: {len(test_targets)}d ✓")

    # Save test for backtest
    with open('test_targets.pkl', 'wb') as f:
        pickle.dump(test_targets.values, f)

    # Scale targets
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_targets.values.reshape(-1,1)).flatten()
    val_scaled = scaler.transform(val_targets.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test_targets.values.reshape(-1,1)).flatten()

    # === 3. GRAPH DATE MAPPING ===
    def get_graph_context(target_date):
        """Get 3 most recent monthly graphs before target_date"""
        available_dates = [pd.Timestamp(d) for d in graph_dates if pd.Timestamp(d) < target_date]
        if len(available_dates) < 3:
            return None
        # Convert Timestamp.date() → str 'YYYY-MM-DD' matching dict keys
        recent_str_dates = [str(d.date()) for d in available_dates[-3:]]
        
        return [monthly_graphs[date_str] for date_str in recent_str_dates]


    # === 4. DYNAMIC DATA LOADERS ===
    def create_dynamic_loader(targets_scaled, mask, graph_context_func):
        data_list = []
        seq_len = 30  # Days of feature history
        
        valid_indices = np.where(mask)[0]
        # Local position counter so we index targets_scaled (split array) correctly
        scaled_pos = 0
        for full_idx in valid_indices:
            if scaled_pos >= seq_len:
                context_graphs = graph_context_func(targets.index[full_idx])
                if context_graphs is not None:
                    # Clone latest graph + update node features from recent days
                    latest_graph = context_graphs[-1].clone()
                    num_nodes = latest_graph.num_nodes

                    # Update node features with recent data (last 30 days before target)
                    recent_features = features[
                        (features.index.get_level_values('Date') >= targets.index[full_idx - seq_len]) &
                        (features.index.get_level_values('Date') < targets.index[full_idx])
                    ]

                    if len(recent_features) > 0:
                        latest_feats = recent_features.groupby('Ticker')[['RV','Skew','Kurt','Vol_Proxy']].last()
                        # Reindex to match graph node order; pad/fill missing nodes
                        graph_nodes = list(latest_graph.x.shape[0:1])  # shape hint only
                        latest_feats = latest_feats.reindex(
                            latest_feats.index[:num_nodes]
                        ).fillna(0.15)
                        # Ensure exactly num_nodes rows
                        if len(latest_feats) < num_nodes:
                            pad = pd.DataFrame(
                                [[0.15] * 4] * (num_nodes - len(latest_feats)),
                                columns=['RV','Skew','Kurt','Vol_Proxy']
                            )
                            latest_feats = pd.concat([latest_feats, pad], ignore_index=True)
                        latest_graph.x = torch.tensor(
                            latest_feats.values[:num_nodes], dtype=torch.float
                        )

                    latest_graph.y = torch.tensor(
                        float(targets_scaled[scaled_pos]), dtype=torch.float
                    )
                    data_list.append(latest_graph)
            scaled_pos += 1
        
        return DataLoader(data_list, batch_size=16, shuffle=True)  # Smaller batch for graphs

    train_loader = create_dynamic_loader(train_scaled, train_mask, get_graph_context)
    val_loader = create_dynamic_loader(val_scaled, val_mask, get_graph_context)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # === 5. TRAINING ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)  # Lower LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=12)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_rmse = float('inf')
    patience_counter = 0

    print("\n=== Training with Dynamic Monthly Graphs ===")
    for epoch in range(250):  # More epochs for complex temporal model
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward: Use graph's own monthly context
            pred = model([batch_data])  # Single graph per batch (dynamic)
            loss = criterion(pred, batch_data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(pred)
            train_count += len(pred)
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0
        train_losses.append(avg_train_loss)

        # Dynamic validation
        model.eval()
        epoch_val_preds, epoch_val_actuals = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                pred = model([batch_data])
                epoch_val_preds.extend(pred.cpu().numpy())
                epoch_val_actuals.extend(batch_data.y.cpu().numpy())
        
        val_rmse = np.sqrt(mean_squared_error(epoch_val_actuals, epoch_val_preds))
        val_losses.append(val_rmse)
        # Keep last epoch's predictions for the scatter plot
        val_preds = epoch_val_preds
        scheduler.step(val_rmse)

        if epoch % 25 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}: Train {avg_train_loss:.4f} | Val RMSE {val_rmse:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), 'gat_dynamic_best.pt')
            print(f"  New best Val RMSE: {val_rmse:.4f}")
        else:
            patience_counter += 1
            if patience_counter > 30:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Best Val RMSE: {best_val_rmse:.4f}")

    # === 6. TEST SET EVALUATION ===
    print("\n=== OOS Test Set ===")
    test_loader = create_dynamic_loader(test_scaled, test_mask, get_graph_context)
    
    model.load_state_dict(torch.load('gat_dynamic_best.pt', weights_only=True))
    model.eval()
    
    test_preds = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model([batch_data])
            test_preds.extend(pred.cpu().numpy())
    
    test_rmse = np.sqrt(mean_squared_error(test_scaled[:len(test_preds)], test_preds))
    print(f"Test RMSE (OOS 2024H2-2026): {test_rmse:.4f}")
    
    # Save for backtest
    test_results = {
        'dates': targets[test_mask].index[:len(test_preds)].tolist(),
        'actual': test_targets.values[:len(test_preds)],
        'predicted': np.array(test_preds),
        'rmse': test_rmse
    }
    with open('dynamic_test_predictions.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    print("Dynamic test predictions saved: dynamic_test_predictions.pkl")

    # === 7. VISUALIZATION ===
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses[:100], label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val RMSE', linewidth=2)
    plt.yscale('log')
    plt.title('Dynamic Graph Training')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(val_scaled[:len(val_preds)], np.array(val_preds), alpha=0.6, s=20)
    plt.plot([val_scaled.min(), val_scaled.max()], [val_scaled.min(), val_scaled.max()], 'r--', lw=2)
    plt.xlabel('Actual (scaled)')
    plt.ylabel('Predicted')
    plt.title(f'Validation\nRMSE: {best_val_rmse:.3f}')
    
    plt.subplot(1, 3, 3)
    plt.scatter(test_scaled[:len(test_preds)], np.array(test_preds), alpha=0.6, s=20, color='green')
    plt.plot([test_scaled.min(), test_scaled.max()], [test_scaled.min(), test_scaled.max()], 'r--', lw=2)
    plt.xlabel('Actual (scaled)')
    plt.ylabel('Predicted')
    plt.title(f'Test OOS\nRMSE: {test_rmse:.3f}')
    
    plt.tight_layout()
    plt.savefig('gat_dynamic_results.png', dpi=300)
    plt.show()

    print("\n✅ Dynamic Monthly GAT Complete!")
    print("Files:")
    print("  - gat_dynamic_best.pt")
    print("  - dynamic_test_predictions.pkl")
    print("  - gat_dynamic_results.png")
