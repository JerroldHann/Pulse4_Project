import streamlit as st
from src.agent import extract_query_info
from src.risk_engine import analyze_risk
from src.graph_tool import render_person_graph
from src.transactions import get_transactions
from src.simulator import save_and_predict

st.set_page_config(page_title="🏦 AI Risk Center", layout="wide")
st.title("🏦 AI Risk Center")

# ---------- Fixed tips ----------
st.markdown("""
### 💡 You can ask:
- **Risk score**: `What's the risk score of transaction T001?`
- **Risk graph**: `Show the relationship graph for account A123`
- **Risk list**: `List all high-risk transactions`
- **Risk analysis**: `Analyze the current risk status`,`List all transactions oF A016`
---
""")

# ---------- Init session state ----------
for key, val in {"intent": "", "auto_name": "", "auto_txid": "", "query": ""}.items():
    st.session_state.setdefault(key, val)

col1, col2 = st.columns([0.45, 0.55])

# ---------- Left: AI Assistant ----------
with col1:
    st.header("💬 AI Assistant")
     # Handle "Clear" state flag before rendering input box
    clear_flag = st.session_state.pop("clear_trigger", False)
    query = st.text_area("Enter your query:", key="input_query")

    if st.button("Analyze"):
        parsed = extract_query_info(query)
        st.json(parsed)
        st.session_state["intent"] = parsed.get("intent", "")
        st.session_state["auto_name"] = parsed.get("name", "")
        st.session_state["auto_txid"] = parsed.get("transaction_id", "")
        st.session_state["query"] = query
        st.toast(f"✅ Parsed: {st.session_state['intent']} → {st.session_state['auto_name']}")

    if st.button("Clear"):
        for k in ["intent", "auto_name", "auto_txid", "input_query"]:
            st.session_state.pop(k, None)
        st.session_state["input_query"] = "" 
        st.session_state["clear_trigger"] = True
        st.rerun()
# ---------- Right: Main Panels ----------
with col2:
    tabs = st.tabs(["⚠️ Risk Score", "🕸 Risk Graph", "📋 Risk Transactions",
    "🧪 Simulated Real-time Data"])

    # Tab 1: Risk Score
    with tabs[0]:
        st.subheader("📊 Risk Score")
        tx = st.text_input(
            "Transaction ID or Client Name",
            key="auto_txid",
            value=st.session_state.get("auto_txid", "")
        )
        if st.button("Run Risk Analysis"):
            st.success(analyze_risk(tx or "T001"))

    # Tab 2: Risk Graph
    with tabs[1]:
        st.subheader("🧩 Risk Graph")
        name = st.text_input(
            "Client Name",
            key="auto_name",
            value=st.session_state.get("auto_name", "")
        )
        if st.button("Generate Graph"):
            html = render_person_graph(name or "A001")
            st.components.v1.html(html, height=600, scrolling=True)
        

    # Tab 3: Risk Transactions
    with tabs[2]:
        st.subheader("📋 Risk Transactions")
        cname = st.text_input("Filter by Client Name", value=st.session_state.get("auto_name", ""))
        min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5)
        df = get_transactions(cname, min_prob)
        st.dataframe(df, use_container_width=True)

    with tabs[3]:
        st.subheader("🧪 Simulated Real-time Data")

        st.markdown("Enter JSON data for simulation:")
        sample_json = """{
            "transaction_id": "T999",
            "orig_account": "A123",
            "dest_account": "B456",
            "amount": 3200.50,
            "fraud_prob": 0.42
        }"""
        user_json = st.text_area("Input JSON:", sample_json, height=200)

        if st.button("Save & Predict"):
            result = save_and_predict(user_json)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result["status"])
                st.json(result["prediction"])


# ---------- Auto-switch tabs based on intent ----------
intent = st.session_state.get("intent", "")
if intent == "risk_graph":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[1].click();</script>", height=0)
elif intent == "risk_list":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[2].click();</script>", height=0)
elif intent == "risk_score":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[0].click();</script>", height=0)
