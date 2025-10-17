import os
import re
import json
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
import datetime
current_time = datetime.datetime.now()
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
        "start_date_time": None,
        "end_date_time": None,
        "confidence": 0.5,
    }


# ========== 4️⃣ Main LLM extraction ==========
def extract_query_info(query: str) -> dict:
    """
    Extract structured query info using Watsonx or fallback.
    Return dict with keys:
    intent, name, transaction_id, merchant_id, start_date_time, end_date_time
    """
    if not _model:
        return _fallback_intent(query)

    prompt = f"""
You are an intent classification assistant for a financial risk analysis system.
Extract structured JSON information from the user query.

Rules:
1️⃣ Return **valid JSON only**, no explanations.
2️⃣ Numeric fields must be integers or null (no strings), and we use current time for computing

3️⃣ If user mentions "today"
 set "start_days_ago"= today 00:00, "end_days_ago"=current 
4️⃣ If user mentions "from X days ago to Y days ago", set accordingly.
5️⃣ If no time mentioned: from today days ago to 31 days ago

Schema:
{{
  "intent": "risk_score" | "risk_graph" | "risk_list" | "other",
  "name": "account name if mentioned, else empty",
  "transaction_id": "transaction id if mentioned, else empty",
  "merchant_id": "",
  "start_date_time": str(YYYY-MM-DD HH:MM:SS),
  "end_date_time": str(YYYY-MM-DD HH:MM:SS)
  "probability_threshold": float (0.0-1.0, default 0.5)
}}

Examples:
If current time is 2025-10-16 20:01:02
User: "What are today's risky transactions?"
→ {{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "",
    "start_date_time": "2025-10-16 00:00:00", "end_date_time": "2025-10-16 20:01:02"}}

User: "Show transactions from 25 days ago to 5 days ago with probability higher than 0.7"
→ {{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "",
    "start_date_time": "2025-09-21 20:01:02", "end_date_time": "2025-10-11 20:01:02","probability_threshold": 0.7}}

User: "Display risk graph for account 241080 over the past week"
→ {{"intent": "risk_graph", "name": "241080", "transaction_id": "", "merchant_id": "",
    "start_date_time": 2025-10-09 20:01:02 "end_date_time": 2025-10-16 20:01:02}}end_date_time is current_time

User: "Display risk graph for account 241080 from 2025-10-09 to 2025-10-16"
→ {{"intent": "risk_graph", "name": "241080", "transaction_id": "", "merchant_id": "",
    "start_date_time": 2025-10-09 00:00:00, "end_date_time": 2025-10-17 00:00:00}}

User: "Display risk graph for account 241080 from 2025-10-09 2am to 2025-10-16 3pm"
→ {{"intent": "risk_graph", "name": "241080", "transaction_id": "", "merchant_id": "",
    "start_date_time": 2025-10-09 02:00:00, "end_date_time": 2025-10-16 15:00:00}}

User: "What's the risk score of transaction T001?"
→ {{"intent": "risk_score", "name": "", "transaction_id": "T001", "merchant_id": "",
     "start_date_time": 2025-9-15 20:01:02, "end_date_time": 2025-10-16 20:01:02}}

current time is the current_time.
Now process this query:
User question: {query,current_time}
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
        parsed.setdefault("start_date_time", None)
        parsed.setdefault("end_date_time", None)
        parsed.setdefault("confidence", 0.9)

        print(f"🔍 Parsed intent: {parsed}")
        return parsed

    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        return _fallback_intent(query)


def risk_score_agent(result):
    """
    Generate a detailed risk report as an explanation from a financial expert, including the formula and extended explanation in English.
    """
    try:
        # Extract input data
        tx_id = result['transaction_id']
        prob = result['input_prob']
        amount = result['amount']
        A0 = result['A0_global']
        RI = result['RI']
        risk_level = result['risk_level'][0]
        explanation = result['explanation'][0]
        recommendation = result['recommendation'][0]
        
        # Risk score formula explanation
        formula = """
        The Composite Risk Index (RI) is calculated as:
        RI = σ1 * P(fraud) + σ2 * Â + σ3 * ln(P/(1-P))
        
        Where:
        - σ1: Weight coefficient for fraud probability (0.6)
        - σ2: Weight coefficient for normalized transaction amount (0.3)
        - σ3: Weight coefficient for logit-transformed probability term (0.1)
        - P(fraud): Probability of fraud
        - Â: Normalized transaction amount (based on global amount percentile A₀)
        - ln(P/(1-P)): Logit transformation
        """
        
        # Generate detailed explanation
        detailed_explanation = f"""
        📊 Transaction {tx_id} Risk Analysis Report:
        
        1. **Fraud Probability (P(fraud))**: {prob[0]:.4f}
        2. **Transaction Amount**: {amount[0]}, with a normalized value (A₀) of {A0:.2f}, meaning this transaction amount is in the top {A0:.2f}% of all historical transactions.
        3. **Risk Score (RI)**: The composite risk score is {RI[0]:.3f}. This score combines the influence of the fraud probability, transaction amount, and the logit-transformed probability term.
        
        Risk Level: **{risk_level}**
        - Explanation: {explanation}
        - Recommendation: {recommendation}
        
        Formula Explanation: {formula}
        """
        
        # Generate a more detailed financial analysis report using the model
        input_text = f"""
        Based on the following transaction data, generate a detailed risk analysis report:

        Transaction ID: {tx_id}
        Fraud Probability: {prob[0]:.4f}
        Transaction Amount: {amount[0]}
        Global Amount Percentile A₀: {A0:.2f}
        Composite Risk Index (RI): {RI[0]:.3f}
        Risk Level: {risk_level}
        Risk Explanation: {explanation}
        Recommended Actions: {recommendation}
        
        Please generate a detailed financial expert analysis for this transaction, explaining the significance of each metric, and providing relevant recommendations based on the risk score.
        """
        
        # Call the model to generate a detailed report, increase max_new_tokens for more content
        if _model:
            response = _model.generate(prompt=input_text, params={"temperature": 0.3, "max_new_tokens": 5000})  # Increase tokens
            print(f"🔍 Full model response: {response}")  # Print the full model response to check
            ai_text = response.get("results")[0]["generated_text"].strip()  # Extract the generated text from the response
        else:
            ai_text = "⚠️ Model not initialized, unable to generate report."
        
        # Return the generated text
        return ai_text
    
    except Exception as e:
        return f"⚠️ Error generating the risk report: {e}"
