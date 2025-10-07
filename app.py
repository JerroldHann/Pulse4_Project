import streamlit as st
from src.agent import extract_query_info
from src.risk_engine import analyze_risk
from src.graph_tool import render_person_graph
from src.transactions import get_transactions

st.set_page_config(page_title="🏦 AI 风控中心", layout="wide")
st.title("🏦 AI 风控中心")

# ---------- 固定提示 ----------
st.markdown("""
### 💡 你可以询问以下内容：
- **风险评分**：`交易T001的风险分数是多少？`  
- **风险图谱**：`显示账户A123的风险关系图`  
- **风险列表**：`列出所有高风险交易`  
- **风险分析**：`分析当前风险状况`
---
""")

# ---------- 初始化状态 ----------
for key, val in {"intent": "", "auto_name": "", "auto_txid": "", "query": ""}.items():
    st.session_state.setdefault(key, val)

col1, col2 = st.columns([0.45, 0.55])

# ---------- 左侧：AI助手 ----------
with col1:
    st.header("💬 AI 助手")
    query = st.text_area("请输入查询：", key="input_query")

    if st.button("Analyze"):
        parsed = extract_query_info(query)
        st.json(parsed)
        st.session_state["intent"] = parsed.get("intent", "")
        st.session_state["auto_name"] = parsed.get("name", "")
        st.session_state["auto_txid"] = parsed.get("transaction_id", "")
        st.session_state["query"] = query
        st.toast(f"✅ 解析成功：{st.session_state['intent']} → {st.session_state['auto_name']}")

    if st.button("🧹 Clear"):
        for k in ["intent", "auto_name", "auto_txid", "input_query"]:
            st.session_state.pop(k, None)
        st.rerun()

# ---------- 右侧：功能区 ----------
with col2:
    tabs = st.tabs(["⚠️ 风险分数", "🕸 风险图谱", "📋 风险交易列表"])

    # Tab 1: 分数
    with tabs[0]:
        st.subheader("📊 风险分数")
        tx = st.text_input("交易编号或客户名", key="auto_txid", value=st.session_state.get("auto_txid", ""))
        if st.button("Run Risk Analysis"):
            st.success(analyze_risk(tx or "T001"))

    # Tab 2: 图谱
    with tabs[1]:
        st.subheader("🧩 风险图谱")
        name = st.text_input("客户名", key="auto_name", value=st.session_state.get("auto_name", ""))
        if st.button("Generate Graph"):
            html = render_person_graph(name or "A001")
            st.components.v1.html(html, height=600, scrolling=True)

    # Tab 3: 列表
    with tabs[2]:
        st.subheader("📋 风险交易列表")
        cname = st.text_input("筛选客户名", value=st.session_state.get("auto_name", ""))
        min_prob = st.slider("最小欺诈概率", 0.0, 1.0, 0.5)
        df = get_transactions(cname, min_prob)
        st.dataframe(df, use_container_width=True)

# ---------- 自动跳转 ----------
intent = st.session_state.get("intent", "")
if intent == "risk_graph":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[1].click();</script>", height=0)
elif intent == "risk_list":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[2].click();</script>", height=0)
elif intent == "risk_score":
    st.components.v1.html("<script>window.parent.document.querySelectorAll('button[role=\"tab\"]')[0].click();</script>", height=0)
