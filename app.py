import streamlit as st
from src.agent import extract_query_info, risk_score_agent

from src.risk_engine import composite_risk_index
from src.graph_tool import render_person_graph, render_high_risk_network
from src.transactions import get_transactions
from src.simulator import save_and_predict
from src.date import date_to_step_range
from src.data_utils import search_prob_amount
import json
import time
from datetime import datetime


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
- **Risk Graph**: Show the risk graph for account 468201 (default all time)
- **Risk Transactions**: List high-risk transactions for account 468201 (default today)
- **Risk Graph**: Show the risk graph for account 468201 over the past week
- **Risk Transactions**: List high-risk transactions from 10 days ago to 5 days ago
---
""")

# ---------- Init session state ----------
for k, v in {
    "intent": "", "auto_name": "", "auto_txid": "",
   "start_date_time": None, "end_date_time": None,
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
            st.session_state["start_date_time"] = parsed.get("start_date_time", "")
            st.session_state["end_date_time"] = parsed.get("end_date_time", "")

    with c2:
        if st.button("Clear", key="btn_clear_auto"):
            for k in ["intent","auto_name","auto_txid",
                      "start_date_time","end_date_time","input_query","use_range"]:
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
    
    # 输入交易ID
    tx_id = st.text_input(
        "Transaction ID",
        key="auto_txid",
        value=_get("auto_txid")
    )
    
    # 如果点击按钮，执行风险分析
    if st.button("Run Risk Analysis", key="btn_run_score"):
        if not tx_id:
            st.error("⚠️ Transaction ID cannot be empty.")
        else:
            # 查询风险得分
            prob,amount=search_prob_amount(tx_id)
            print(f"Query result: prob={prob}, amount={amount}")
            result = composite_risk_index(
                prob=[prob],  # 默认欺诈概率，或根据实际场景传入
                amount=[amount],  # 默认交易金额，或根据实际场景传入
                transaction_id=tx_id,
                verbose=True
            )
            
            # 输出结果
            st.success("✅ Risk score calculation completed")
            ai_text = risk_score_agent(result)
            st.write(ai_text) 

# === Tab 2: Risk Graph ===
with tabs[1]:
    
    st.subheader("🧩 Risk Graph")
    name = st.text_input("Client Name", key="auto_name2", value=_get("auto_name2", ""))
    role = st.selectbox("Role filter", ["both", "origin", "destination"], index=0, key="role_graph")
    
    start_date_time_auto = st.session_state.get("start_date_time")
    end_date_time_auto = st.session_state.get("end_date_time")
    
    # Check if start_date_time_auto is not None before parsing
    if start_date_time_auto:
        start_date_time_auto = datetime.strptime(start_date_time_auto, "%Y-%m-%d %H:%M:%S")
        start_date = start_date_time_auto.date()
        start_time = start_date_time_auto.time()
    else:
        start_date = datetime.now().date()  # Default to today's date if None
        start_time = datetime.now().time()  # Default to current time if None
    
    # Check if end_date_time_auto is not None before parsing
    if end_date_time_auto:
        end_date_time_auto = datetime.strptime(end_date_time_auto, "%Y-%m-%d %H:%M:%S")
        end_date = end_date_time_auto.date()
        end_time = end_date_time_auto.time()
    else:
        end_date = datetime.now().date()  # Default to today's date if None
        end_time = datetime.now().time()  # Default to current time if None

    # 用户选择开始日期和时间
    start_date = st.date_input("Start date", min_value=datetime(2025, 9, 15),  value=start_date)
    start_time = st.time_input("Start time", value=start_time)

    # 用户选择结束日期和时间
    end_date = st.date_input("End date", min_value=datetime(2025, 9, 15),  value=end_date)
    end_time = st.time_input("End time", value=end_time)
    
    # 将日期和时间拼接为 datetime 对象
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    # 将 datetime 对象转换为 "YYYY-MM-DD HH:MM:SS" 格式
    start_date_time_auto = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    end_date_time_auto = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    if st.button("Generate Graph", key="btn_graph"):
        # 判断是否使用日期范围
        step_range = date_to_step_range(start_date_time_auto, end_date_time_auto)
        
        # 生成图形的 HTML 内容
        html = render_person_graph(name or "241080", role=role, step_range=step_range)
        
        # 使用 Streamlit 组件显示生成的 HTML 文件
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
