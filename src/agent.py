import os, re, json
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

_model = None
if WATSONX_API_KEY and WATSONX_PROJECT_ID:
    try:
        _model = Model(
            model_id="ibm/granite-3-2-8b-instruct",
            params={"temperature": 0.2, "max_new_tokens": 200},
            credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
        )
        print("✅ IBM Watsonx AI model initialized.")
    except Exception as e:
        print(f"⚠️ Failed to init Watsonx model: {e}")
else:
    print("⚠️ Missing Watsonx credentials, rule-based fallback will be used.")


def _fallback_intent(query: str) -> dict:
    """Rule-based fallback when LLM fails."""
    q = query.lower()
    intent = "other"
    if any(k in q for k in ["graph", "network", "relation", "visual", "plot", "图", "网络", "关系"]):
        intent = "risk_graph"
    elif any(k in q for k in ["list", "which", "all", "清单", "列表"]):
        intent = "risk_list"
    elif any(k in q for k in ["score", "prob", "grade", "分数", "概率"]):
        intent = "risk_score"
    elif any(k in q for k in ["analysis", "trend", "report", "统计", "分析"]):
        intent = "risk_analysis"

    name = ""
    txid = ""
    # e.g. A123 / 241080 etc.
    acc = re.search(r"[A-Za-z]*\d{3,}", query)
    if acc:
        name = acc.group()
    # e.g. T001
    tx = re.search(r"T\d+", query, re.IGNORECASE)
    if tx:
        txid = tx.group()

    return {
        "intent": intent,
        "name": name,
        "transaction_id": txid,
        "merchant_id": "",
        "days_ago": 0,
        "start_days_ago": None,
        "end_days_ago": None,
        "confidence": 0.5,
    }


def extract_query_info(query: str) -> dict:
    """
    Use watsonx to extract:
      intent in {"risk_score","risk_graph","risk_list","other"},
      name, transaction_id, merchant_id,
      days_ago or start_days_ago/end_days_ago (integer offsets).
    """
    if not _model:
        return _fallback_intent(query)

    prompt = f"""
You are an intent classification assistant for a financial risk analysis system.
Extract structured JSON from the user query. Return JSON only.

Schema:
{{
  "intent": "risk_score" | "risk_graph" | "risk_list" | "other",
  "name": "account name if mentioned, else empty",
  "transaction_id": "transaction id if mentioned, else empty",
  "merchant_id": "",
  "days_ago": 0,
  "start_days_ago": null,
  "end_days_ago": null
}}

Examples:
User: "What are today's risky transactions?"
{{"intent":"risk_list","name":"","transaction_id":"","merchant_id":"","days_ago":0,"start_days_ago":null,"end_days_ago":null}}

User: "Show risky transactions from 10 days ago to 5 days ago"
{{"intent":"risk_list","name":"","transaction_id":"","merchant_id":"","days_ago":0,"start_days_ago":10,"end_days_ago":5}}

User: "Display risk graph for account 241080 over the past week"
{{"intent":"risk_graph","name":"241080","transaction_id":"","merchant_id":"","days_ago":0,"start_days_ago":7,"end_days_ago":0}}

User question: {query}
"""

    try:
        resp = _model.generate(prompt=prompt)
        text = resp["results"][0]["generated_text"].strip()

        # strip code fences
        text = text.replace("```json", "").replace("```", "").strip()
        i, j = text.find("{"), text.rfind("}") + 1
        if i == -1 or j <= 0:
            print("⚠️ LLM returned non-JSON:", text)
            return _fallback_intent(query)

        parsed = json.loads(text[i:j])
        # normalize fields
        parsed.setdefault("intent", "other")
        parsed.setdefault("name", "")
        parsed.setdefault("transaction_id", "")
        parsed.setdefault("merchant_id", "")
        parsed.setdefault("days_ago", 0)
        parsed.setdefault("start_days_ago", None)
        parsed.setdefault("end_days_ago", None)
        parsed.setdefault("confidence", 0.9)
        return parsed
    except Exception as e:
        print("⚠️ LLM parsing failed:", e)
        return _fallback_intent(query)
