import streamlit as st
from dotenv import load_dotenv
from src.agent import ask_agent, extract_query_info
from src.graph_tool import render_person_graph
from src.risk_engine import analyze_risk

load_dotenv()
st.set_page_config(page_title="🏦 Bank Risk AI Agent", layout="wide")

st.title("🏦 Bank Risk Intelligence Dashboard")

col1, col2 = st.columns([0.45, 0.55])

# ============================
# 左侧：AI 对话区
# ============================
with col1:
    st.header("💬 AI Risk Assistant")
    query = st.text_area("Ask about a client or transaction:")

    if st.button("Analyze"):
        with st.status("🤖 Interpreting your request...", expanded=True) as status:
            parsed = extract_query_info(query)
            st.write("**AI parsed your query as:**")
            st.json(parsed)
            status.update(label="✅ Interpretation complete", state="complete")
        st.session_state["parsed"] = parsed  # 缓存结果

# ============================
# 右侧：智能交互区
# ============================
with col2:
    tab1, tab2 = st.tabs(["⚠️ Risk Analysis", "🕸 Transaction Graph"])

    # 风险分析
    with tab1:
        st.subheader("📊 Risk Score Panel")
        name = st.text_input("Client Name", value=st.session_state.get("parsed", {}).get("name", ""))
        tx_id = st.text_input("Transaction ID", value=st.session_state.get("parsed", {}).get("transaction_id", ""))

        if not name and not tx_id:
            st.info("💡 Tip: Ask AI about a client or transaction to auto-fill fields.")
        elif st.button("Run Risk Analysis"):
            result = analyze_risk(name or tx_id)
            st.success(result)

    # 关系图
    with tab2:
        st.subheader("🧩 Transaction Relationship Graph")
        graph_name = st.text_input("Client (or group) Name", value=st.session_state.get("parsed", {}).get("name", ""))

        if not graph_name:
            st.info("💡 Tip: Ask AI for a client relationship to auto-fill name field.")
        elif st.button("Generate Graph"):
            html = render_person_graph(graph_name)
            st.components.v1.html(html, height=600, scrolling=True)
