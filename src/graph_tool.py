import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
from .data_utils import resolve_today_csv, load_data_by_days_ago

def _to_str(x):  # safe cast
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)

def _build_edges_df(days_range=None) -> pd.DataFrame:
    """
    Load one or multiple daily CSVs and concatenate.
    days_range: None -> today; (start_days_ago, end_days_ago) inclusive descending.
    """
    if not days_range:
        csv_path = resolve_today_csv()
        return pd.read_csv(csv_path)
    start, end = int(days_range[0]), int(days_range[1])
    frames = []
    for d in range(start, end - 1, -1):
        try:
            frames.append(load_data_by_days_ago(d))
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def render_person_graph(
    client_name: str,
    role: str = "both",  # "both" | "origin" | "destination"
    days_range=None,
    output_html="data/graph.html"
) -> str:
    """
    Build a 1-hop neighborhood graph for a given account with role filtering.
    """
    df = _build_edges_df(days_range)
    if df.empty:
        return "<p>⚠️ No data available for the selected date range.</p>"

    need = ["orig_id", "dest_id", "transaction_id", "fraud_prob_pred"]
    if any(c not in df.columns for c in need):
        return "<p>⚠️ CSV missing required columns.</p>"

    df["orig_id"] = df["orig_id"].apply(_to_str)
    df["dest_id"] = df["dest_id"].apply(_to_str)
    client = str(client_name)

    if role == "origin":
        sub = df[df["orig_id"] == client]
    elif role == "destination":
        sub = df[df["dest_id"] == client]
    else:
        sub = df[(df["orig_id"] == client) | (df["dest_id"] == client)]

    if sub.empty:
        return f"<p>⚠️ No transactions for {client} with role={role}.</p>"

    G = nx.Graph()
    for _, r in sub.iterrows():
        G.add_edge(
            r["orig_id"], r["dest_id"],
            tx=str(r["transaction_id"]),
            prob=float(r["fraud_prob_pred"])
        )

    if client not in G:
        return f"<p>⚠️ {client} not present in the graph.</p>"

    # 1-hop ego graph
    subG = nx.ego_graph(G, client, radius=1)

    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    # Nodes (no border by default, visible border when selected)
    for n in subG.nodes():
        node_size = 18
        net.add_node(
            n,
            color="#444444" if n == client else "#A0A0A0",
            label=n,
            size=node_size,
            borderWidth=0,
            borderWidthSelected=max(2, node_size // 10),
        )

    # Edges
    for u, v, d in subG.edges(data=True):
        color = "red" if d.get("prob", 0) > 0.7 else "gray"
        net.add_edge(
            u, v,
            title=f"Transaction: {d.get('tx','N/A')} | Fraud Probability={d.get('prob',0):.2f}",
            color=color
        )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.save_graph(output_html)
    with open(output_html, encoding="utf-8") as f:
        return f.read()

def render_high_risk_network(df: pd.DataFrame, output_html: str) -> str:
    """
    Render a global risk network from a filtered dataframe (e.g., high-risk list).
    """
    if df.empty:
        return "<p>⚠️ No high-risk transactions to visualize.</p>"

    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(
            _to_str(r["orig_id"]),
            _to_str(r["dest_id"]),
            tx=str(r["transaction_id"]),
            prob=float(r.get("fraud_prob_pred", 0))
        )

    net = Network(height="650px", width="100%", bgcolor="#ffffff")
    for n in G.nodes():
        net.add_node(n, label=n, color="#5B8FF9", size=14, borderWidth=0, borderWidthSelected=2)
    for u, v, d in G.edges(data=True):
        color = "red" if d.get("prob", 0) > 0.7 else "gray"
        net.add_edge(u, v, color=color, title=f"{d.get('tx')} | prob={d.get('prob',0):.2f}")

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    net.save_graph(output_html)
    with open(output_html, encoding="utf-8") as f:
        return f.read()
