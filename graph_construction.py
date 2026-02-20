import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import torch
from torch_geometric.data import Data
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Load features
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([
    features.index.levels[0], 
    pd.to_datetime(features.index.levels[1])
])
features = features.sort_index()

print(f"Features: {features.shape}, Tickers: {len(features.index.unique('Ticker'))}")

# === ROLLING MONTHLY GRAPHS ===
window_corr = 60
granger_window = 252
corr_threshold = 0.7
granger_p = 0.05

# Monthly dates (1st of each month)
monthly_dates = pd.date_range(start='2015-01-01', end='2026-01-01', freq='MS')  
monthly_dates = monthly_dates[monthly_dates.isin(features.index.get_level_values('Date').unique())]

central_tickers = [
    "^VIX", "SPY", "QQQ", "XLE", "XLK", "XLF", 
    "CL=F", "TLT", "GC=F", "^TNX"
]

print(f"Building {len(monthly_dates)} monthly graphs...")
print(f"Using {len(central_tickers)} vol leaders for Granger tests")

monthly_graphs = {}  # {date: graph_data}

for month_end in monthly_dates[6:]:  # Skip first 6 months (warmup)
    print(f"Building graph for {month_end.date()}...")
    
    # Data up to month_end (no future leak)
    features_up_to = features[features.index.get_level_values('Date') < month_end]
    if len(features_up_to) < window_corr + granger_window:
        continue
        
    rv_data = features_up_to['RV'].unstack('Ticker').dropna()
    
    # === STEP 1: CORRELATION EDGES ===
    rv_pivot = rv_data.tail(window_corr * 2)  # Recent 120 days for stability
    if len(rv_pivot) < window_corr:
        continue
        
    rolling_corr = rv_pivot.rolling(window_corr).corr().dropna()
    if rolling_corr.empty:
        continue
        
    # Latest available corr snapshot BEFORE month_end
    available_dates = rolling_corr.index.get_level_values(0).unique()
    corr_date = available_dates[available_dates < month_end][-1]
    corr_snapshot = rolling_corr.loc[corr_date]
    
    corr_edges = []
    corr_snapshot_abs = corr_snapshot.abs()
    for i in corr_snapshot_abs.index:
        for j in corr_snapshot_abs.columns:
            if i != j:
                weight = corr_snapshot_abs.loc[i, j]
                if weight > corr_threshold:
                    corr_edges.append({
                        'source': i, 'target': j, 
                        'weight': float(weight), 'type': 'corr'
                    })
    
    # === STEP 2: GRANGER EDGES ===
    granger_edges = []
    rv_recent = rv_data.tail(granger_window)
    if len(rv_recent) > 100:
        for i in central_tickers:
            if i in rv_recent.columns:
                for j in rv_recent.columns:
                    if i != j and j in rv_recent.columns:
                        try:
                            test_data = rv_recent[[i, j]].dropna()
                            if len(test_data) > 60:
                                gc_res = grangercausalitytests(
                                    test_data[[j, i]], maxlag=4, verbose=False
                                )
                                pvals = [res[0]['ssr_ftest'][1] for res in gc_res.values()]
                                min_p = min(pvals)
                                if min_p < granger_p:
                                    granger_edges.append({
                                        'source': i, 'target': j,
                                        'weight': float(1 - min_p),
                                        'type': 'granger',
                                        'lag': int(np.argmin(pvals) + 1)
                                    })
                        except:
                            pass
    
    # === STEP 3: NETWORKX GRAPH ===
    G = nx.DiGraph()
    all_tickers = list(rv_data.columns)
    G.add_nodes_from(all_tickers)
    
    # Corr edges (bidirectional)
    corr_df = pd.DataFrame(corr_edges)
    if not corr_df.empty:
        for _, edge in corr_df.iterrows():
            G.add_edge(edge['source'], edge['target'], **edge)
            G.add_edge(edge['target'], edge['source'], **edge)
    
    # Granger edges (directed)
    granger_df = pd.DataFrame(granger_edges)
    if not granger_df.empty:
        for _, edge in granger_df.iterrows():
            G.add_edge(edge['source'], edge['target'], **edge)
    
    # === STEP 4: PYTORCH GEOMETRIC ===
    if G.number_of_edges() < 50:  # Skip sparse graphs
        continue
        
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    edge_index_list, edge_attr_list = [], []
    for u, v, d in G.edges(data=True):
        edge_index_list.append([node_to_idx[u], node_to_idx[v]])
        edge_attr_list.append([d['weight'], 1.0 if d['type'] == 'corr' else 0.0])
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    # Node features: MOST RECENT before month_end
    node_feat_df = (
        features_up_to.groupby('Ticker')[['RV', 'Skew', 'Kurt', 'Vol_Proxy']]
        .last().reindex(list(G.nodes()))
        .fillna(0.15)
    )
    node_features = torch.tensor(node_feat_df.values, dtype=torch.float)
    
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=G.number_of_nodes(),
        date=str(month_end.date())  # Metadata
    )
    
    monthly_graphs[str(month_end.date())] = graph_data
    print(f"  ✓ {month_end.date()}: {G.number_of_nodes()}n/{G.number_of_edges()}e")

# === SAVE ROLLING GRAPHS ===
torch.save(monthly_graphs, 'monthly_graphs.pt')
print(f"\n✅ Saved {len(monthly_graphs)} monthly graphs to monthly_graphs.pt")

# Summary stats
edge_counts = [g.num_edges for g in monthly_graphs.values()]
print(f"Graphs range: {min(edge_counts)}-{max(edge_counts)} edges")
print(f"Avg edges/graph: {np.mean(edge_counts):.0f}")
