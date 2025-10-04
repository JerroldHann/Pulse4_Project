from pyvis.network import Network

def render_graph():
    net = Network(height="600px", width="100%", bgcolor="#111", font_color="white")
    net.barnes_hut()

    net.add_node("A", label="Client A", color="#FFD700", title="Medium Risk")
    net.add_node("B", label="Client B", color="#FF4500", title="High Risk")
    net.add_node("C", label="Client C", color="#32CD32", title="Low Risk")
    net.add_edge("A", "B", title="Transaction: $50,000")
    net.add_edge("B", "C", title="Transaction: $8,000")

    return net.generate_html()
