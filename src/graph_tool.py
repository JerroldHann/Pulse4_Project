import networkx as nx
from pyvis.network import Network
import pandas as pd

def render_person_graph(client_name: str, csv_path="data/daily_transactions.csv"):
    """
    Generate and render an interactive transaction graph for a specific client.
    """
    # Load transaction data
    df = pd.read_csv(csv_path)

    # Build network graph
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row["orig_account"], 
            row["dest_account"],
            tx=row["transaction_id"], 
            prob=row["fraud_prob"]
        )

    # Handle missing client
    if client_name not in G:
        return "<p>⚠️ No transaction records found for this client.</p>"

    # Build a 1-hop neighborhood subgraph
    subG = nx.ego_graph(G, client_name, radius=1)

    # Initialize visualization
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    
    # Add nodes with colors (orange for center, green for neighbors)
    for n in subG.nodes():
        node_size = 20
        net.add_node(
            n,
            color="#444444" if n == client_name else "#A0A0A0",
            label=n,
            size=node_size,                 # smaller nodes
            borderWidth=0,           # no border before click
            borderWidthSelected=node_size/10   # 2 border on click
        )

    # Add edges (red for high-risk, gray otherwise)
    for u, v, d in subG.edges(data=True):
        color = "red" if d["prob"] > 0.7 else "gray"
        net.add_edge(
            u, v,
            title=f"Transaction: {d['tx']} | Fraud Probability = {d['prob']:.2f}",
            color=color
        )

    # Save and return HTML
    net.save_graph("data/graph.html")
    with open("data/graph.html", encoding="utf-8") as f:
        return f.read()
