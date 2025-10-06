from pyvis.network import Network
import random
import os

def render_person_graph(name: str):
    """
    模拟生成交易关联图
    """
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    nodes = [name] + [f"Client-{i}" for i in range(1, 5)]
    for n in nodes:
        net.add_node(n, label=n)
    for i in range(1, len(nodes)):
        net.add_edge(name, nodes[i], label=f"Tx {random.randint(100,999)}")
    path = "data/temp_graph.html"
    os.makedirs("data", exist_ok=True)
    net.save_graph(path)
    return open(path, "r", encoding="utf-8").read()
