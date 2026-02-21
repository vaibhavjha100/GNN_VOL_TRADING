#!/usr/bin/env python3
"""
display_graphs.py - FIXED Timelapse (matplotlib animation scope bug resolved)

PyTorch 2.6+ compatible | Scope-fixed | Interactive + Export
"""

import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PyTorch 2.6+ SAFE LOADING
try:
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
    print("✅ PyTorch Geometric Data safe-loaded")
except ImportError:
    print("⚠️  torch_geometric not found")

def load_monthly_graphs(file_path='monthly_graphs.pt'):
    """Load PyG graphs safely."""
    monthly_graphs = torch.load(file_path, weights_only=False, map_location='cpu')
    dates = sorted(monthly_graphs.keys())
    graphs = [monthly_graphs[date] for date in dates]
    print(f"✅ Loaded {len(graphs)} monthly graphs: {dates[0]} → {dates[-1]}")
    return graphs, dates

def data_to_nx(data):
    """PyG Data → NetworkX DiGraph."""
    G = nx.DiGraph()
    ticker_order = data.ticker_order
    
    # Nodes with RV features
    for idx, row in enumerate(data.x):
        ticker = ticker_order[idx]
        G.add_node(ticker, rv=float(row[0]), size=max(200, 300 + row[0]*1500))
    
    # Edges
    for i in range(data.edge_index.size(1)):
        src_idx, tgt_idx = data.edge_index[:, i]
        src, tgt = ticker_order[src_idx], ticker_order[tgt_idx]
        weight, is_corr = data.edge_attr[i]
        edge_type = 'corr' if is_corr else 'granger'
        G.add_edge(src, tgt, weight=float(weight), type=edge_type)
    
    return G

def compute_fixed_positions(graphs):
    """Fixed spring layout."""
    all_tickers = set()
    for data in graphs:
        all_tickers.update(data.ticker_order)
    
    union_g = nx.DiGraph()
    union_g.add_nodes_from(all_tickers)
    for data in graphs[:3]:
        union_g.add_edges_from(data_to_nx(data).edges())
    
    print(f"Layout for {len(all_tickers)} tickers...")
    return nx.spring_layout(union_g, k=2, iterations=100, seed=42)

def main(save_mode=None, file_path='monthly_graphs.pt'):
    graphs, dates = load_monthly_graphs(file_path)
    if not graphs:
        return
    
    pos = compute_fixed_positions(graphs)
    central_tickers = ["^VIX", "SPY", "QQQ", "CL=F", "TLT", "GC=F", "^TNX"]
    
    # FIXED: Proper closure capturing
    def make_update_func():
        def update(frame):
            G = data_to_nx(graphs[frame])
            
            ax.clear()
            
            # Static nodes
            static_nodes = set(pos) - set(G.nodes())
            if static_nodes:
                nx.draw_networkx_nodes(nx.DiGraph(), pos, nodelist=list(static_nodes),
                                     node_color='lightgray', node_size=80, alpha=0.2, ax=ax)
            
            # Active nodes
            rv_values = [G.nodes[n]['rv'] for n in G.nodes()]
            sizes = [G.nodes[n]['size'] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=rv_values, cmap=plt.cm.Reds,
                                 node_size=sizes, vmin=0, vmax=0.5, edgecolors='black',
                                 linewidths=0.8, ax=ax)
            
            # Corr edges (blue)
            corr_edges = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='corr']
            if corr_edges:
                weights = [G[u][v]['weight']*8 for u,v in corr_edges]
                nx.draw_networkx_edges(G, pos, edgelist=corr_edges, width=weights,
                                     edge_color='steelblue', alpha=0.8, arrows=False, ax=ax)
            
            # Granger edges (red arrows)
            granger_edges = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='granger']
            if granger_edges:
                nx.draw_networkx_edges(G, pos, edgelist=granger_edges, arrows=True,
                                     arrowstyle='->', arrowsize=20, edge_color='darkred',
                                     alpha=0.9, width=2, ax=ax)
            
            # Central labels
            labels = {t: t for t in central_tickers if t in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
            
            # Title
            rv_min, rv_max = min(rv_values), max(rv_values)
            ax.set_title(f"Asset Network Timelapse\n"
                        f"{dates[frame]} | {G.number_of_nodes()}n/{G.number_of_edges()}e | "
                        f"RV: {rv_min:.1%}–{rv_max:.1%}", fontsize=14, pad=20)
            ax.axis('off')
            ax.set_facecolor('#f8f9fa')
        
        return update
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#f8f9fa')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    
    # Animation
    update_func = make_update_func()
    ani = FuncAnimation(fig, update_func, frames=len(graphs), interval=1200, 
                       repeat=True, blit=False)
    
    if save_mode == 'mp4':
        writer = FFMpegWriter(fps=0.8, bitrate=3000)
        filename = f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M')}.mp4"
        print(f"💾 Exporting {filename}...")
        ani.save(filename, writer=writer)
        print("✅ MP4 ready!")
        plt.close()
    
    elif save_mode == 'gif':
        writer = PillowWriter(fps=1)
        ani.save("timelapse.gif", writer=writer)
        print("✅ GIF ready!")
        plt.close()
    
    else:
        print("▶️  Timelapse playing... (Close window to stop)")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', choices=['mp4', 'gif'])
    parser.add_argument('--file', default='monthly_graphs.pt')
    args = parser.parse_args()
    main(args.save, args.file)
