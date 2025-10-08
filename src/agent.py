import os
import re
import json
from ibm_watsonx_ai.foundation_models import Model
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# ----------------------------
# Initialize the model
# ----------------------------
model = None
if WATSONX_API_KEY and WATSONX_PROJECT_ID:
    try:
        model = Model(
            model_id="meta-llama/llama-2-13b-chat",
            params={"temperature": 0.2, "max_new_tokens": 200},
            credentials={"apikey": WATSONX_API_KEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
        )
        print("✅ IBM Watsonx AI model initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to initialize model: {e}")
else:
    print("⚠️ Missing Watsonx credentials, running in fallback mode.")


# ----------------------------
# Query understanding with LLM
# ----------------------------
def extract_query_info(query: str):
    """
    Use the Watsonx model to intelligently extract the user's intent,
    name, and transaction information from natural language input.
    """
    prompt = f"""
You are a professional AI assistant for financial risk control.
Return **only** a JSON object, without any extra explanation.

Task:
Analyze the user's financial risk-related query, identify the intent,
and extract key information.

Output format (must be valid JSON):
{{
  "intent": "risk_score" or "risk_graph" or "risk_list" or "risk_analysis" or "other",
  "name": "account name",
  "transaction_id": "transaction ID", 
  "merchant_id": "merchant ID",
  "confidence": 0.95
}}

Intent definitions:
- "risk_score": Asking for a risk score / probability / value.
- "risk_graph": Requesting a visualization / graph / relationship map / network.
- "risk_list": Listing high-risk transactions or suspicious activities.
- "risk_analysis": Requesting analysis / statistics / trend / report.
- "other": Any other type of query.

Important: Return JSON only. No explanations, no text outside braces.

Examples:
User: Show me the risk score of account A123
{{"intent": "risk_score", "name": "A123", "transaction_id": "", "merchant_id": "", "confidence": 0.95}}

User: Show today's risk transaction graph
{{"intent": "risk_graph", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.9}}

User: Analyze risk trends
{{"intent": "risk_analysis", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.85}}

User: List all high-risk transactions
{{"intent": "risk_list", "name": "", "transaction_id": "", "merchant_id": "", "confidence": 0.9}}

Now process this user query:
User: {query}
"""
    try:
        response = model.generate(prompt=prompt)
        text = response["results"][0]["generated_text"]
        print(f"🔍 Raw model output: {text}")

        # Clean up and extract JSON
        json_text = text.strip().replace("```json", "").replace("```", "")
        start_idx, end_idx = json_text.find("{"), json_text.rfind("}") + 1
        parsed = json.loads(json_text[start_idx:end_idx])

        # Ensure required fields exist
        for key in ["intent", "name", "transaction_id", "merchant_id", "confidence"]:
            parsed.setdefault(key, "")
        return parsed

    except Exception as e:
        print(f"⚠️ Query parsing failed: {e}")
        return fallback_intent_recognition(query)


# ----------------------------
# Fallback intent recognition (rule-based)
# ----------------------------
def fallback_intent_recognition(query: str):
    """Fallback rule-based intent extraction when LLM parsing fails."""
    query_lower = query.lower()
    result = {
        "intent": "other",
        "name": "",
        "transaction_id": "",
        "merchant_id": "",
        "confidence": 0.5,
    }

    # Intent detection based on keywords
    if any(k in query_lower for k in ["graph", "network", "visualization", "relation", "map", "plot"]):
        result["intent"] = "risk_graph"
    elif any(k in query_lower for k in ["list", "show all", "which", "records"]):
        result["intent"] = "risk_list"
    elif any(k in query_lower for k in ["score", "value", "probability", "rating"]):
        result["intent"] = "risk_score"
    elif any(k in query_lower for k in ["analysis", "trend", "report", "statistics", "overview"]):
        result["intent"] = "risk_analysis"

    # Extract possible account name (like A123)
    acc = re.search(r"[A-Za-z]+\d+", query)
    if acc:
        result["name"] = acc.group()

    # Extract transaction ID (like T001)
    tx = re.search(r"T\d+", query)
    if tx:
        result["transaction_id"] = tx.group()

    print(f"🔧 Fallback intent recognition result: {result}")
    return result
