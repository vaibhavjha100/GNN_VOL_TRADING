import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

# Load features (MultiIndex: Ticker, Date)
features = pd.read_csv("features.csv", index_col=['Ticker', 'Date'])
features.index = features.index.set_levels([features.index.levels[0],
                                            pd.to_datetime(features.index.levels[1])])
features = features.sort_index()

print(f"Features: {features.shape}, Tickers: {len(features.index.unique('Ticker'))}")

# === STEP 1: CORRELATION EDGES (Rolling 60-day) ===
window_corr = 60
rv_pivot = features['RV'].unstack('Ticker').dropna()

# Rolling correlation matrix
rolling_corr = rv_pivot.rolling(window_corr).corr().dropna()

# Use only the last date's correlation snapshot (most recent 60-day window)
last_date = rolling_corr.index.get_level_values(0).unique()[-1]
corr_snapshot = rolling_corr.loc[last_date]

# Extract edges: |corr| > 0.5 threshold
corr_edges = []
corr_snapshot_abs = corr_snapshot.abs()
for i in corr_snapshot_abs.index:
    for j in corr_snapshot_abs.columns:
        if i != j:
            weight = corr_snapshot_abs.loc[i, j]
            if weight > 0.5:
                corr_edges.append({'source': i, 'target': j, 'weight': float(weight), 'type': 'corr'})

corr_df = pd.DataFrame(corr_edges)
print(f"Correlation edges: {len(corr_df)} (threshold 0.5)")

# === STEP 2: GRANGER CAUSALITY EDGES (last 252 days) ===
granger_edges = []
rv_data = features['RV'].unstack('Ticker').dropna()
rv_recent = rv_data.iloc[-252:]  # Use last 1yr of data for efficiency

for i in rv_recent.columns[:10]:  # Test first 10 tickers as source (scale later)
    for j in rv_recent.columns:
        if i != j:
            try:
                test_data = rv_recent[[i, j]].dropna()
                if len(test_data) > 50:
                    gc_res = grangercausalitytests(test_data[[j, i]], maxlag=5, verbose=False)
                    pvals = [res[0]['ssr_ftest'][1] for res in gc_res.values()]
                    min_p = min(pvals)
                    if min_p < 0.05:
                        granger_edges.append({
                            'source': i, 'target': j,
                            'weight': float(1 - min_p),  # Strength: 1 - p-value
                            'type': 'granger',
                            'lag': int(np.argmin(pvals) + 1)
                        })
            except Exception:
                pass

granger_df = pd.DataFrame(granger_edges)
print(f"Granger edges: {len(granger_df)} (p<0.05)")

# === STEP 3: COMBINE EDGES (NetworkX DiGraph) ===
G = nx.DiGraph()

# Add all ticker nodes
all_tickers = list(rv_data.columns)
G.add_nodes_from(all_tickers)

# Add correlation edges (undirected → bidirectional)
if not corr_df.empty:
    for _, edge in corr_df.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'], type=edge['type'])
        G.add_edge(edge['target'], edge['source'], weight=edge['weight'], type=edge['type'])

# Add Granger edges (directed: cause → effect)
if not granger_df.empty:
    for _, edge in granger_df.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'], type=edge['type'])

print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("Top 10 edges:")
for u, v, d in list(G.edges(data=True))[:10]:  # list() to allow slicing
    print(f"  {u} → {v}: {d['weight']:.3f} ({d['type']})")

# === STEP 4: PYTORCH GEOMETRIC FORMAT (GAT Ready) ===
# Build node → integer index mapping
node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

edge_index_list = []
edge_attr_list = []

for u, v, d in G.edges(data=True):
    edge_index_list.append([node_to_idx[u], node_to_idx[v]])
    edge_attr_list.append([d['weight'], 1.0 if d['type'] == 'corr' else 0.0])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float)

# Node features: latest available feature vector per ticker (ordered by node_to_idx)
node_feat_df = (
    features
    .groupby('Ticker')[['RV', 'Skew', 'Kurt', 'Vol_Proxy']]
    .last()
    .reindex(list(G.nodes()))   # ensure same order as node_to_idx
)
node_features = torch.tensor(node_feat_df.values, dtype=torch.float)

graph_data = Data(
    x=node_features,        # N_tickers × 4
    edge_index=edge_index,  # 2 × N_edges
    edge_attr=edge_attr,    # N_edges × 2  (weight, type flag)
    num_nodes=G.number_of_nodes()
)

print(f"\nPyG Data Ready:")
print(f"  Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}")
print(f"  Node feature shape : {graph_data.x.shape}")
print(f"  Edge index shape   : {graph_data.edge_index.shape}")
print(f"  Sample edge attr   : {graph_data.edge_attr[:3]}")

# Save for GAT training
torch.save(graph_data, 'volatility_graph.pt')
print("\n✅ Graph saved: volatility_graph.pt")
