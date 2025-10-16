import streamlit as st
from src.agent import extract_query_info
from src.risk_engine import analyze_risk
from src.graph_tool import render_person_graph, render_high_risk_network
from src.transactions import get_transactions
from src.simulator import save_and_predict
import json
import time


# ---------- Basic Setup ----------
st.set_page_config(page_title="🏦 AI Risk Center", layout="wide")
st.title("🏦 AI Risk Center")

# ---------- Helper ----------
def _get(key, default=""):
    return st.session_state.get(key, default)

def _set(key, value):
    st.session_state[key] = value

def json_dumps_pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

# ---------- Tips ----------
st.markdown("""
### 💡 You can ask:
- **Risk Score**: What's the risk score of transaction 1735197544?
- **Risk Graph**: Show the risk graph for account 468201 (default today)
- **Risk Transactions**: List high-risk transactions for account 468201 (default today)
- **Todo: **
- **Risk Graph**: Show the risk graph for account 468201 over the past week
- **Risk Transactions**: List high-risk transactions from 10 days ago to 5 days ago
---
""")

# ---------- Init session state ----------
for k, v in {
    "intent": "", "auto_name": "", "auto_txid": "",
    "days_ago": None, "start_days_ago": None, "end_days_ago": None,
    "use_range": False, "input_query": "", "sim_json": ""
}.items():
    st.session_state.setdefault(k, v)

col1, col2 = st.columns([0.45, 0.55])

# ---------- Left: AI Assistant ----------
with col1:
    st.header("💬 AI Assistant")

    query = st.text_area("Enter your query:", key="input_query")

    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        if st.button("Analyze", key="btn_analyze"):
            parsed = extract_query_info(query)
            st.json(parsed)

            # --- Extract values ---
            st.session_state["intent"] = parsed.get("intent", "")
            st.session_state["auto_name"] = parsed.get("name", "")
            st.session_state["auto_name2"] = parsed.get("name", "")
            st.session_state["auto_name3"] = parsed.get("name", "")
            st.session_state["auto_txid"] = parsed.get("transaction_id", "")
            st.session_state["query"] = query
            # _set("intent", parsed.get("intent", ""))
            # _set("auto_name", parsed.get("name", ""))
            # _set("auto_txid", parsed.get("transaction_id", ""))
            
            days_ago = parsed.get("days_ago", None)
            start_days = parsed.get("start_days_ago", None)
            end_days = parsed.get("end_days_ago", None)

            # --- Unified time logic ---
            if days_ago is not None:
                _set("days_ago", days_ago)
                _set("start_days_ago", days_ago)
                _set("end_days_ago", days_ago)
                st.session_state["use_range"] = True
                #_set("use_range", True)
            elif start_days is not None or end_days is not None:
                _set("days_ago", None)
                _set("start_days_ago", start_days)
                _set("end_days_ago", end_days)
                st.session_state["use_range"] = True
                # _set("use_range", True)
            else:
                _set("days_ago", None)
                _set("start_days_ago", None)
                _set("end_days_ago", None)
                st.session_state["use_range"] = False
                # _set("use_range", False)

            st.toast(f"✅ Parsed: {_get('intent')} → {_get('auto_name') or _get('auto_txid')}")

    with c2:
        if st.button("Clear", key="btn_clear_auto"):
            for k in ["intent","auto_name","auto_txid","days_ago",
                      "start_days_ago","end_days_ago","input_query","use_range"]:
                st.session_state.pop(k, None)
            st.session_state["clear_trigger"] = True
            st.rerun()

# ---------- Right: Panels ----------
with col2:
    tabs = st.tabs([
        "⚠️ Risk Score",
        "🕸 Risk Graph",
        "📋 Risk Transactions",
        "🧪 Simulated Real-time Data"
    ])

    # === Tab 1: Risk Score ===
    with tabs[0]:
        st.subheader("📊 Risk Score")
        tx = st.text_input(
            "Transaction ID",
            key="auto_txid",
            value=_get("auto_txid")
        )
        if st.button("Run Risk Analysis", key="btn_run_score"):
            st.success(analyze_risk(tx or "T001"))

    # === Tab 2: Risk Graph ===
    with tabs[1]:
        st.subheader("🧩 Risk Graph")
        name = st.text_input("Client Name", key="auto_name2", value=_get("auto_name", ""))
        role = st.selectbox("Role filter", ["both", "origin", "destination"], index=0, key="role_graph")
        use_range = st.checkbox(
            "Use date range (days ago)",
            value=any([
                _get("use_range", False),
                _get("days_ago") is not None,
                _get("start_days_ago") is not None,
                _get("end_days_ago") is not None
            ]),
            key="chk_use_range_graph"
        )

        start_d = st.number_input(
            "Start days ago", min_value=0,
            value=int(_get("start_days_ago") or 0),
            step=1, key="num_start_graph",
            disabled=not use_range
        )
        end_d = st.number_input(
            "End days ago", min_value=0,
            value=int(_get("end_days_ago") or 0),
            step=1, key="num_end_graph",
            disabled=not use_range
        )

        if st.button("Generate Graph", key="btn_graph"):
            days_range = (int(start_d), int(end_d)) if use_range else None
            html = render_person_graph(name or "241080", role=role, days_range=days_range)
            st.components.v1.html(html, height=600, scrolling=True)

    # === Tab 3: Risk Transactions ===
    with tabs[2]:
        st.subheader("📋 Risk Transactions")
        cname = st.text_input("Filter by Client Name", value=_get("auto_name",""), key="auto_name3")
        min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, key="sld_prob")
        colA, colB, colC = st.columns(3)
        with colA:
            use_range2 = st.checkbox(
                "Use date range (days ago)",
                
                value=any([
                    _get("use_range", False),
                    _get("days_ago") is not None,
                    _get("start_days_ago") is not None,
                    _get("end_days_ago") is not None
                ]),
                key="chk_use_range_list"
            )
        with colB:
            s2 = st.number_input(
                "Start days ago", min_value=0,
                value=int(_get("start_days_ago") or 0),
                step=1, key="num_start_list",
                disabled=not use_range2
            )
        with colC:
            e2 = st.number_input(
                "End days ago", min_value=0,
                value=int(_get("end_days_ago") or 0),
                step=1, key="num_end_list",
                disabled=not use_range2
            )

        df = get_transactions(
            client_name=cname,
            min_prob=min_prob,
            start_days_ago=int(s2) if use_range2 else None,
            end_days_ago=int(e2) if use_range2 else None
        )
        st.dataframe(df, use_container_width=True, key="df_list")

        if st.button("Build High-Risk Network", key="btn_highrisk"):
            html_name = (
                f"risk_network_{int(s2)}to{int(e2)}days.html"
                if use_range2 else
                f"risk_network_{int(_get('days_ago',0))}daysago.html"
            )
            html = render_high_risk_network(df, output_html=html_name)
            st.components.v1.html(html, height=600, scrolling=True)

    # === Tab 4: Simulated Real-time Data ===
    with tabs[3]:
        st.subheader("🧪 Simulated Real-time Data")
        st.markdown("Paste one JSON object per submission:")

        sample = {
            "step": 710,
            "orig_id": "88640",
            "dest_id": "60",
            "amount": 1000.0,
            "orig_old_balance": 10000.0,
            "orig_new_balance": 9000.0,
            "dest_old_balance": 4000.0,
            "dest_new_balance": 5000.0
        }
        sim_text = st.text_area(
            "Input JSON", key="txt_sim_json",
            value=_get("sim_json") or json_dumps_pretty(sample), height=180
        )

        colX, colY = st.columns([0.5, 0.5])
        with colX:
            if st.button("Save & Predict", key="btn_sim_save"):
                result = save_and_predict(sim_text)
                
                    
        with colY:
            if st.button("Clear input", key="btn_clear_sim"):
                st.session_state.pop("sim_json", None)
                st.rerun()

# ---------- Auto-switch tabs ----------
intent = _get("intent","")
if intent == "risk_graph":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[1].click();</script>", height=0)
elif intent == "risk_list":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[2].click();</script>", height=0)
elif intent == "risk_score":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[0].click();</script>", height=0)
