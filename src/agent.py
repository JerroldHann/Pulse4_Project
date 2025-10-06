import os
import json
import re
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

model = None
if WATSONX_API_KEY and WATSONX_PROJECT_ID:
    try:
        model = Model(
            model_id="ibm/granite-3-2-8b-instruct",
            params={"temperature": 0.2, "max_new_tokens": 200},
            credentials={"url": WATSONX_URL, "apikey": WATSONX_API_KEY},
            project_id=WATSONX_PROJECT_ID,
        )
        print("✅ IBM Watsonx AI model initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to initialize Watsonx model: {e}")
else:
    print("⚠️ Watsonx credentials not found. Running in offline mode.")


def extract_query_info(query: str):
    """
    使用 Watsonx 提取意图、姓名、交易编号。
    支持模糊输出清洗与回退。
    """
    if not model:
        intent = "graph" if "图" in query else "risk" if "风险" in query else "other"
        name_match = re.search(r"[\u4e00-\u9fa5]{2,3}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", query)
        name = name_match.group(0) if name_match else ""
        return {"intent": intent, "name": name, "transaction_id": ""}

    prompt = f"""
     
        请从以下用户问题中提取关键信息，仅返回 JSON：
        
        字段：
        - intent: 判断用户想找的是risk score还是risk graph还是other，其中如果用户risk score/risk graph这两个都想找就填other
        - name: 人名（如有），中文或者英文名都可以，作为语言学家判断是不是名字
        - transaction_id: 交易编号（如有），如果用户填了交易编号

        问题: {query}

        示例输出：
        {{"intent": "", "name": "", "transaction_id": ""}}
        """

    try:
        response = model.generate(prompt=prompt)
        text = response["results"][0]["generated_text"]
        print("🧠 Raw model output:", text)

        # 截取第一个 JSON 块
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            print("🧠 Parsed:", data)
            return data
        else:
            print("⚠️ No JSON found, fallback to regex.")
    except Exception as e:
        print("❌ extract_query_info error:", e)

    # fallback: 简单关键词推测
    intent = "graph" if "图" in query else "risk" if "风险" in query else "other"
    name_match = re.search(r"[\u4e00-\u9fa5]{2,3}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", query)
    name = name_match.group(0) if name_match else ""
    return {"intent": intent, "name": name, "transaction_id": ""}

def ask_agent(query: str) -> str:
    """兼容旧接口"""
    return "🤖 Agent ready — use the right panel for details."


if __name__ == "__main__":
    test = "查一下张力的风险"
    print(extract_query_info(test))
