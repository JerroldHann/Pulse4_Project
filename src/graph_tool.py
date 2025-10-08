import networkx as nx
from pyvis.network import Network
import pandas as pd

def render_person_graph(client_name: str, csv_path="data/daily_transactions.csv"):
    """
    Generate and render an interactive transaction graph for a specific client.
    """
    df = pd.read_csv(csv_path)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row["orig_account"], 
            row["dest_account"],
            tx=row["transaction_id"], 
            prob=row["fraud_prob"]
        )

    if client_name not in G:
        return "<p>⚠️ No transaction records found for this client.</p>"

    subG = nx.ego_graph(G, client_name, radius=1)

    net = Network(height="600px", width="100%", bgcolor="#ffffff")

    for n in subG.nodes():
        node_size = 20
        net.add_node(
            n,
            color="#444444" if n == client_name else "#A0A0A0",
            label=n,
            size=node_size,
            borderWidth=0,
            borderWidthSelected=node_size/10
        )

    for u, v, d in subG.edges(data=True):
        color = "red" if d["prob"] > 0.7 else "gray"
        net.add_edge(
            u, v,
            title=f"Transaction: {d['tx']} | Fraud Probability = {d['prob']:.2f}",
            color=color
        )

    # Save as temporary HTML
    html_path = "data/graph.html"
    net.save_graph(html_path)

    # ✅ Inject JavaScript listener (minimal modification)
    js_script = """
    <script>
    // Wait for vis network to load
    window.addEventListener('load', () => {
        const network = window.network;
        if (!network) return;
        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                const nodeName = params.nodes[0];
                console.log("Clicked node:", nodeName);
                // Send message to Streamlit parent iframe
                window.parent.postMessage({type: "node_click", node: nodeName}, "*");
            }
        });
    });
    </script>
    """

    # Append JS to HTML file
    with open(html_path, "a", encoding="utf-8") as f:
        f.write(js_script)

    with open(html_path, encoding="utf-8") as f:
        return f.read()
