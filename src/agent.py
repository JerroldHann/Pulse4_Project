import os, re, json
from ibm_watsonx_ai.foundation_models import Model
from dotenv import load_dotenv

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

model = None
if WATSONX_API_KEY and WATSONX_PROJECT_ID:
    try:
        model = Model(
            model_id="meta-llama/llama-2-13b-chat",
            params={"temperature": 0.2, "max_new_tokens": 200},
            credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
        )
        print("✅ IBM Watsonx AI model initialized.")
    except Exception as e:
        print(f"⚠️ Failed to init model: {e}")


def extract_query_info(query: str):
    """
    使用 watsonx 模型智能识别自然语言查询意图。
    """
    prompt = f"""
你是一个专业的金融风控AI助手。请只返回JSON格式，不要任何额外解释。

任务：
分析用户输入的金融风控相关问题，识别意图并提取关键信息。

输出格式（必须是有效的JSON）：
{{
  "intent": "risk_score" 或 "risk_graph" 或 "risk_list" 或 "risk_analysis" 或 "other",
  "name": "账户名",
  "transaction_id": "交易编号", 
  "merchant_id": "商户编号",
  "confidence": 0.95
}}

意图定义：
- "risk_score": 查询风险评分、分数、得分
- "risk_graph": 请求可视化、图、图谱、网络、关系图
- "risk_list": 列出高风险交易、风险清单
- "risk_analysis": 风险分析、统计、趋势、报告
- "other": 其他请求

重要：只返回JSON，不要其他任何文字！

示例：
用户：查看A123账户的风险分数
{{"intent": "risk_score", "name": "A123", "transaction_id": "", "merchant_id": "", "confidence": 0.95}}

用户：显示今天的风险交易图谱  
{{"intent": "risk_graph", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.9}}

用户：分析风险趋势
{{"intent": "risk_analysis", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.85}}

用户：列出高风险交易
{{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.9}}

现在处理这个查询：
用户：{query}
"""
    try:
        response = model.generate(prompt=prompt)
        text = response["results"][0]["generated_text"]
        print(f"🔍 模型原始输出: {text}")

        json_text = text.strip()
        json_text = json_text.replace("```json", "").replace("```", "")
        start_idx, end_idx = json_text.find("{"), json_text.rfind("}") + 1
        parsed = json.loads(json_text[start_idx:end_idx])

        for key in ["intent", "name", "transaction_id", "merchant_id", "confidence"]:
            parsed.setdefault(key, "")
        return parsed
    except Exception as e:
        print(f"⚠️ 意图识别失败: {e}")
        return fallback_intent_recognition(query)


def fallback_intent_recognition(query: str):
    query_lower = query.lower()
    result = {
        "intent": "other",
        "name": "",
        "transaction_id": "",
        "merchant_id": "",
        "confidence": 0.5
    }
    if any(k in query_lower for k in ["图", "图谱", "网络", "关系", "可视化"]):
        result["intent"] = "risk_graph"
    elif any(k in query_lower for k in ["列表", "清单", "哪些", "全部"]):
        result["intent"] = "risk_list"
    elif any(k in query_lower for k in ["分数", "得分", "评分"]):
        result["intent"] = "risk_score"
    elif any(k in query_lower for k in ["分析", "报告", "趋势", "统计"]):
        result["intent"] = "risk_analysis"
    acc = re.search(r"[A-Za-z]+\d+", query)
    if acc:
        result["name"] = acc.group()
    tx = re.search(r"T\d+", query)
    if tx:
        result["transaction_id"] = tx.group()
    print(f"🔧 使用备用识别: {result}")
    return result
