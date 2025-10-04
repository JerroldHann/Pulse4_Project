import streamlit as st
from dotenv import load_dotenv
from src.agent import ask_agent
from src.graph_tool import render_graph
from src.visualizer import risk_radar

load_dotenv()
st.set_page_config(page_title="🏦 Bank Risk AI Agent", layout="wide")

st.title("🏦 Bank Risk Intelligence Dashboard")

col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.header("💬 AI Risk Assistant")
    query = st.text_area("Ask about a client or transaction:")
    if st.button("Analyze"):
        with st.status("Analyzing risk...", expanded=True) as status:
            answer = ask_agent(query)
            st.markdown(f"**AI:** {answer}")
            status.update(label="✅ Analysis complete!", state="complete")

with col2:
    tab1, tab2 = st.tabs(["🕸 Network Graph", "📊 Risk Radar"])
    with tab1:
        html = render_graph()
        st.components.v1.html(html, height=600, scrolling=True)
    with tab2:
        risk_radar()
