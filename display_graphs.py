#!/usr/bin/env python3
"""
financial_network_PRO.py - CLEAN 100n/4880e visualization

✅ Edge threshold • Stable layout • Edge/node labels • Types/weights visible
"""

import torch
from pyvis.network import Network
import networkx as nx
import webbrowser
import os

try:
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
except:
    pass

def data_to_nx(data, corr_threshold=0.75):  # FILTER weak edges
    G = nx.DiGraph()
    ticker_order = data.ticker_order
    
    # Nodes
    for idx, row in enumerate(data.x):
        ticker = ticker_order[idx]
        G.add_node(ticker, rv=float(row[0]), size=20 + row[0]*100)
    
    # EDGES: Only strong correlations
    strong_edges = 0
    for i in range(data.edge_index.size(1)):
        src_idx, tgt_idx = data.edge_index[:, i]
        src, tgt = ticker_order[src_idx], ticker_order[tgt_idx]
        weight, is_corr = data.edge_attr[i]
        
        if weight >= corr_threshold:  # Only |corr| > 0.75
            edge_type = 'corr' if is_corr else 'granger'
            G.add_edge(src, tgt, weight=float(weight), type=edge_type)
            strong_edges += 1
    
    print(f"   → Filtered to {strong_edges} strong edges (|r|>{corr_threshold})")
    return G

def visualize_pro_graph(file_path='monthly_graphs.pt', date_idx=0):
    monthly_graphs = torch.load(file_path, weights_only=False, map_location='cpu')
    dates = sorted(monthly_graphs.keys())
    date_str = dates[date_idx]
    data = monthly_graphs[date_str]
    
    G = data_to_nx(data, corr_threshold=0.8)  # Tune 0.7-0.85
    
    # Pyvis PRO
    net = Network(height="950px", width="1600px", directed=False, 
                  notebook=False, cdn_resources='in_line')
    
    # CENTRAL TICKERS bigger/labeled
    central = [
    "^VIX", "SPY", "QQQ", "XLE", "XLK", "XLF", 
    "CL=F", "TLT", "GC=F", "^TNX"
    ]
    for node_id, node_data in G.nodes(data=True):
        rv = node_data['rv']
        size = 35 + rv*120 if node_id in central else 20 + rv*80
        label = node_id if node_id in central else ""
        
        net.add_node(node_id,
                    size=size,
                    color=f"hsl({int(240-200*rv)}, 70%, 55%)",
                    title=f"{node_id}<br><b>RV: {rv:.1%}</b>",
                    label=label,
                    font={"size": 12 if node_id in central else 0,
                          "color": "#000"})
    
    # COLORED EDGES with weights
    corr_edges, granger_edges = 0, 0
    for src, tgt, edge_data in G.edges(data=True):
        weight = edge_data['weight']
        if edge_data['type'] == 'corr':
            color = "#2196F3"  # Blue
            corr_edges += 1
        else:
            color = "#F44336"  # Red
            granger_edges += 1
        
        net.add_edge(src, tgt,
                    width=2 + weight*8,
                    color=color,
                    dashes=False,
                    title=f"<b>{edge_data['type'].upper()}</b><br>"
                          f"Weight: {weight:.3f}")
    
    print(f"   → {corr_edges} blue corr / {granger_edges} red granger edges")
    
    # FIXED LAYOUT: No chaotic physics
    net.set_options("""
    {
        "physics": {
            "enabled": false
        },
        "layout": {
            "randomSeed": 42,
            "hierarchical": false
        },
        "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "selectConnectedEdges": true,
            "zoomView": true,
            "dragNodes": true
        },
        "edges": {
            "font": {
                "align": "middle"
            }
        },
        "nodes": {
            "font": {
                "size": 11
            }
        }
    }
    """)
    
    filename = f"financial_network_PRO_{date_str.replace('-','_')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(net.generate_html())
    
    webbrowser.open(f"file://{os.path.abspath(filename)}")
    print(f"✅ CLEAN NETWORK: {filename}")
    print("🎯 Central tickers labeled • Hover weights • Drag/zoom • No physics chaos!")

if __name__ == "__main__":
    visualize_pro_graph()
