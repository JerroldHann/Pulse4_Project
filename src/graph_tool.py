import networkx as nx
from pyvis.network import Network
import pandas as pd

def render_person_graph(client_name: str, csv_path="data/daily_transactions.csv"):
    df = pd.read_csv(csv_path)
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r["orig_account"], r["dest_account"],
                   tx=r["transaction_id"], prob=r["fraud_prob"])
    if client_name not in G:
        return "<p>⚠️ 未找到该客户的交易记录</p>"
    subG = nx.ego_graph(G, client_name, radius=1)
    net = Network(height="600px", width="100%", bgcolor="#111")
    for n in subG.nodes():
        net.add_node(n, color="#FF6B35" if n==client_name else "#10B981", label=n)
    for u, v, d in subG.edges(data=True):
        color = "red" if d["prob"]>0.7 else "gray"
        net.add_edge(u, v, title=f"{d['tx']} 概率={d['prob']:.2f}", color=color)
    net.save_graph("data/graph.html")
    with open("data/graph.html", encoding="utf-8") as f:
        return f.read()
