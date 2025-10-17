import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np

from .data_utils import resolve_today_csv, load_data_by_days_ago

def _to_str(x):  # safe cast
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def classify_fraud_patterns(G, risk_threshold=0.5):
    """
    仅对涉及高风险交易（fraud_prob_pred > threshold）的节点进行欺诈类型分类。
    其他节点保持 Normal，不参与分类。
    """
    print("\n🔍 基于风险交易子图识别欺诈类型...")

    # 1️⃣ 提取所有风险交易边
    risky_edges = [(u, v) for u, v, data in G.edges(data=True)
                if data.get("fraud_prob_pred", 0) > risk_threshold or data.get("fraud", False)]
    risky_nodes = set([n for edge in risky_edges for n in edge])

    if not risky_nodes:
        print("⚠️ 没有检测到风险交易，跳过分类。")
        return {}

    # 2️⃣ 构建风险子图
    subG = G.subgraph(risky_nodes).copy()

    # 3️⃣ 计算结构特征
    degree_dict = dict(subG.degree())
    clustering_dict = nx.clustering(subG)
    avg_degree = np.mean(list(degree_dict.values()))
    std_degree = np.std(list(degree_dict.values()))

    labels = {}

    # === F1: 星型欺诈 ===
    for n, deg in degree_dict.items():
        if deg > avg_degree + 2 * std_degree:
            labels[n] = "F1_Star_Fraud"

    # === F4: 孤立欺诈 ===
    for n, deg in degree_dict.items():
        if deg == 1:
            labels[n] = "F4_Isolated_Pair"

    # === F3: 循环欺诈 ===
    try:
        cycles = nx.cycle_basis(subG)
        for cycle in cycles:
            for n in cycle:
                if n not in labels:
                    labels[n] = "F3_Cycle_Fraud"
    except Exception:
        pass

    # === F2: 链式欺诈 ===
    for n, deg in degree_dict.items():
        if deg == 2 and n not in labels:
            labels[n] = "F2_Chain_Fraud"

    # === F5: 团伙欺诈 ===
    communities = [c for c in nx.connected_components(subG) if len(c) > 5]
    for comm in communities:
        avg_clust = np.mean([clustering_dict.get(n, 0) for n in comm])
        if avg_clust > 0.6:
            for n in comm:
                if n not in labels:
                    labels[n] = "F5_Community_Fraud"

    # === 为风险节点添加标签 ===
    nx.set_node_attributes(G, {n: labels.get(n, "Risk_Node") for n in risky_nodes}, "fraud_type")

    print(f"✅ 已为 {len(risky_nodes)} 个风险节点添加欺诈类型标签")
    return labels
def _build_edges_df(step_range) -> pd.DataFrame:
    """
    Load one or multiple daily CSVs and concatenate.
    step_range: (step_start, step_end) inclusive descending.
    """
    csv_path = 'data/test_predictions_v2.0.csv' 
    df = pd.read_csv(csv_path)
    print(step_range)
    if not step_range:
        # 没写就读全部
        return df
    start, end = sorted((int(step_range[0]), int(step_range[1])), reverse=False)
    # 过滤条件：step >= 11 且 step <= 200
    filtered_df = df[(df['step'] >= start) & (df['step'] <= end)]
    return filtered_df

def render_person_graph(
    client_name: str,
    role: str = "both",  # "both" | "origin" | "destination"
    step_range=None,
    output_html="data/risk_graph.html"
) -> str:
    """
    Build a 1-hop neighborhood graph for a given account with role filtering.
    """
    df = _build_edges_df(step_range)
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
        color = "red" if d.get("prob", 0) > 0.5 else "gray"
        net.add_edge(
            u, v,
            title=f"Transaction: {d.get('tx','N/A')} | Fraud Probability={d.get('prob',0):.2f}",
            color=color
        )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    net.save_graph(output_html)
    with open(output_html, encoding="utf-8") as f:
        return f.read()


def render_high_risk_network(df: pd.DataFrame, output_html: str = "risk_network.html", risk_threshold: float = 0.5) -> str:
    """
    构建并绘制包含欺诈类型分类的全局风险网络。
    数据来自过滤后的交易 DataFrame。
    """
    if df.empty:
        return "<p>⚠️ No high-risk transactions to visualize.</p>"

    # === 1️⃣ 构建交易网络 ===
    G = nx.Graph()
    for _, r in df.iterrows():
        orig = str(r["orig_id"])
        dest = str(r["dest_id"])
        prob = float(r.get("fraud_prob_pred", 0))
        tx = str(r.get("transaction_id", "N/A"))
        amt = float(r.get("amount", 0))
        G.add_edge(orig, dest,
                   fraud_prob_pred=prob,
                   tx_id=tx,
                   amount=amt,
                   fraud=(prob > risk_threshold))

    # === 2️⃣ 分类欺诈模式 ===
    labels = classify_fraud_patterns(G, risk_threshold=risk_threshold)

    # === 3️⃣ 可视化配置 ===
    net = Network(height="700px", width="100%", bgcolor="#ffffff", directed=False)
    net.force_atlas_2based(gravity=-40)

    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "font": {"size": 14}
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": false
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)

    # === 4️⃣ 按欺诈类型上色 ===
    color_map = {
        "F1_Star_Fraud": "#ff4d4d",
        "F2_Chain_Fraud": "#00b050",
        "F3_Cycle_Fraud": "#0070c0",
        "F4_Isolated_Pair": "#808080",
        "F5_Community_Fraud": "#9b59b6",
        "Risk_Node": "#ff9900",
        "Normal": "#5B8FF9"
    }

    for n in G.nodes():
        fraud_type = G.nodes[n].get("fraud_type", "Normal")
        degree = G.degree(n)
        node_color = color_map.get(fraud_type, "#5B8FF9")
        net.add_node(
            n,
            label=n,
            color=node_color,
            size=15 + degree * 1.2,
            title=f"Account: {n}<br>Fraud Type: {fraud_type}<br>Degree: {degree}"
        )

    # === 5️⃣ 添加边信息 ===
    for u, v, d in G.edges(data=True):
        prob = d.get("fraud_prob_pred", 0)
        color = "red" if prob > 0.8 else ("orange" if prob > 0.5 else "gray")
        width = 3 if prob > 0.8 else (2 if prob > 0.5 else 1)
        tx = d.get("tx_id", "N/A")
        amt = d.get("amount", 0)
        net.add_edge(
            u, v,
            color=color,
            width=width,
            title=f"Transaction: {tx}<br>Amount: {amt:.2f}<br>Fraud Prob: {prob:.3f}"
        )

    # === 6️⃣ 输出结果 ===
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    net.save_graph(output_html)

    with open(output_html, encoding="utf-8") as f:
        return f.read()
    
