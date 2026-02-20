"""
Temporal GAT Volatility Predictor - FULL PRODUCTION PIPELINE
Trains on your volatility_graph.pt → Predicts tomorrow's portfolio vol
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TemporalGAT(torch.nn.Module):
    def __init__(self, in_channels=4, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden*heads, hidden, heads=1, dropout=dropout)
        self.temporal = torch.nn.GRU(hidden, hidden, batch_first=True)  # Time component
        self.fc = torch.nn.Linear(hidden, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # GAT layers (graph attention)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        
        # Global mean pool (portfolio aggregation)
        x_pool = global_mean_pool(x, batch)
        
        # Temporal GRU (sequence memory)
        x_gru, _ = self.temporal(x_pool.unsqueeze(0))
        x_final = x_gru.squeeze(0)
        
        return self.fc(x_final).squeeze()

def create_time_series_data(graph_data, targets_scaled, seq_len=30):
    data_list = []
    # Use CPU for list construction to avoid memory issues, will move batches to device in loop
    for i in range(seq_len, len(targets_scaled)):
        batch_data = graph_data.clone().cpu()
        batch_data.y = torch.tensor(targets_scaled[i], dtype=torch.float)
        data_list.append(batch_data)
    return DataLoader(data_list, batch_size=32, shuffle=True)

if __name__ == "__main__":
    print("=== Phase 4: GAT Training ===")

    # === 1. LOAD GRAPH + FEATURES ===
    graph_data = torch.load('volatility_graph.pt', weights_only=False)
    print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    # Load features for targets (RV tomorrow)
    features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
    features.index = features.index.set_levels([features.index.levels[0], 
                                              pd.to_datetime(features.index.levels[1])])
    features = features.sort_index()

    # Real targets: Tomorrow's RV (portfolio average)
    rv_wide = features['RV'].unstack('Ticker').dropna()
    targets = rv_wide.shift(-1).mean(axis=1).dropna()  # Portfolio vol tomorrow
    print(f"Targets: {len(targets)} days, mean {targets.mean():.3f}")

    # Time-series split (no future leak)
    split_date = '2025-01-01'
    train_mask = targets.index < split_date
    train_targets = targets[train_mask].values
    val_targets = targets[~train_mask].values

    scaler = StandardScaler()
    train_targets_scaled = scaler.fit_transform(train_targets.reshape(-1,1)).flatten()

    # === 3. TRAINING LOOP ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = torch.nn.MSELoss()

    graph_data = graph_data.to(device)
    train_losses, val_losses = [], []

    train_loader = create_time_series_data(graph_data, train_targets_scaled)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = criterion(pred, batch_data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stability
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation (holdout period)
        model.eval()
        with torch.no_grad():
            val_graph = graph_data.to(device)
            val_pred = model(val_graph)
            val_loss = criterion(val_pred, torch.tensor(val_targets.mean(), device=device))
        
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train {train_losses[-1]:.4f}, Val {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'gat_model_best.pt')
        else:
            patience_counter += 1
            if patience_counter > 20:
                print("Early stopping")
                break

    print(f"✅ Best Val Loss: {best_val:.4f}")
    torch.save(model.state_dict(), 'gat_model_final.pt')

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.yscale('log')
    plt.legend()
    plt.title('GAT Training Convergence')
    plt.savefig('gat_training_curve.png')
    plt.show()

    print("=== PREDICTION EXAMPLE ===")
    model.eval()
    with torch.no_grad():
        test_graph = graph_data.to(device)
        portfolio_vol_pred = model(test_graph).cpu().item()
        portfolio_vol_actual = targets.iloc[-1]  # Latest known
        
    print(f"Predicted portfolio vol:  {portfolio_vol_pred*100:.2f}%")
    print(f"Actual (yesterday):       {portfolio_vol_actual*100:.2f}%")
    print(f"Error:                    {abs(portfolio_vol_pred - portfolio_vol_actual):.3f}")

    # Per-node predictions (stock-level signals)
    node_emb = model.conv2(model.conv1(test_graph.x, test_graph.edge_index), 
                          test_graph.edge_index)
    high_vol_nodes = torch.topk(node_emb.mean(dim=1), k=5).indices
    print("\nTop 5 High Vol Predicted Stocks:")
    tickers = list(test_graph.nodes) if hasattr(test_graph, 'nodes') else [f"Node_{i}" for i in range(test_graph.num_nodes)]
    for i in high_vol_nodes:
        print(f"  {tickers[i]}")
