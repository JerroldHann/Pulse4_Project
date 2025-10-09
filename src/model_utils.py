# src/model_utils.py
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

def get_watsonx_model():
    api = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    pid = os.getenv("WATSONX_PROJECT_ID")
    if not api or not pid:
        return None
    try:
        return Model(
            model_id="meta-llama/llama-2-13b-chat",
            params={"temperature": 0.2, "max_new_tokens": 200},
            credentials={"apikey": api, "url": url},
            project_id=pid,
        )
    except Exception:
        return None
