import os
import re
import json
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model

# ========== 1️⃣ Load env ==========
load_dotenv()
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# ========== 2️⃣ Initialize model ==========
_model = None
if WATSONX_API_KEY and WATSONX_PROJECT_ID:
    try:
        _model = Model(
            model_id="ibm/granite-3-2-8b-instruct",
            params={"temperature": 0.2, "max_new_tokens": 250},
            credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
        )
        print("✅ IBM Watsonx AI model initialized.")
    except Exception as e:
        print(f"⚠️ Failed to initialize Watsonx model: {e}")
else:
    print("⚠️ Missing Watsonx credentials, fallback rules will be used.")


# ========== 3️⃣ Fallback rules ==========
def _fallback_intent(query: str) -> dict:
    """Rule-based fallback when LLM fails or offline."""
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

    # extract IDs
    name, txid = "", ""
    acc = re.search(r"[A-Za-z]*\d{3,}", query)
    if acc:
        name = acc.group()
    tx = re.search(r"T\d+", query, re.IGNORECASE)
    if tx:
        txid = tx.group()

    return {
        "intent": intent,
        "name": name,
        "transaction_id": txid,
        "merchant_id": "",
        "days_ago": None,
        "start_days_ago": None,
        "end_days_ago": None,
        "confidence": 0.5,
    }


# ========== 4️⃣ Main LLM extraction ==========
def extract_query_info(query: str) -> dict:
    """
    Extract structured query info using Watsonx or fallback.
    Return dict with keys:
    intent, name, transaction_id, merchant_id, days_ago, start_days_ago, end_days_ago.
    """
    if not _model:
        return _fallback_intent(query)

    prompt = f"""
You are an intent classification assistant for a financial risk analysis system.
Extract structured JSON information from the user query.

Rules:
1️⃣ Return **valid JSON only**, no explanations.
2️⃣ Numeric fields must be integers or null (no strings).
3️⃣ If user mentions a single day like "today", "yesterday", or "5 days ago":
    → set "days_ago" to that integer, and both "start_days_ago" and "end_days_ago" to null.
4️⃣ If user mentions a range like "from 10 days ago to 5 days ago":
    → set "days_ago" to null, "start_days_ago"=10, "end_days_ago"=5.
5️⃣ If no time mentioned: all three = null.

Schema:
{{
  "intent": "risk_score" | "risk_graph" | "risk_list" | "other",
  "name": "account name if mentioned, else empty",
  "transaction_id": "transaction id if mentioned, else empty",
  "merchant_id": "",
  "days_ago": integer or null,
  "start_days_ago": integer or null,
  "end_days_ago": integer or null
}}

Examples:
User: "What are today's risky transactions?"
→ {{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "",
    "days_ago": 0, "start_days_ago": null, "end_days_ago": null}}

User: "Show risky transactions from 10 days ago to 5 days ago"
→ {{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "",
    "days_ago": null, "start_days_ago": 10, "end_days_ago": 5}}

User: "Display risk graph for account 241080 over the past week"
→ {{"intent": "risk_graph", "name": "241080", "transaction_id": "", "merchant_id": "",
    "days_ago": null, "start_days_ago": 7, "end_days_ago": 0}}

User: "What's the risk score of transaction T001?"
→ {{"intent": "risk_score", "name": "", "transaction_id": "T001", "merchant_id": "",
    "days_ago": null, "start_days_ago": null, "end_days_ago": null}}

Now process this query:
User question: {query}
"""

    try:
        resp = _model.generate(prompt=prompt)
        text = resp["results"][0]["generated_text"].strip()

        # cleanup & extract JSON
        text = text.replace("```json", "").replace("```", "").strip()
        i, j = text.find("{"), text.rfind("}") + 1
        if i == -1 or j <= 0:
            print("⚠️ LLM returned non-JSON:", text)
            return _fallback_intent(query)

        parsed = json.loads(text[i:j])

        # normalize missing keys
        parsed.setdefault("intent", "other")
        parsed.setdefault("name", "")
        parsed.setdefault("transaction_id", "")
        parsed.setdefault("merchant_id", "")
        parsed.setdefault("days_ago", None)
        parsed.setdefault("start_days_ago", None)
        parsed.setdefault("end_days_ago", None)
        parsed.setdefault("confidence", 0.9)

        print(f"🔍 Parsed intent: {parsed}")
        return parsed

    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        return _fallback_intent(query)
