import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model

load_dotenv()

def init_model():
    api = os.getenv("WATSONX_API_KEY")
    pid = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api or not pid:
        print("⚠️ Watsonx credentials missing, using offline mode.")
        return None

    model = Model(
        model_id="meta-llama/llama-2-13b-chat",
        params={"temperature": 0.2, "max_new_tokens": 200},
        credentials={"apikey": api, "url": url},
        project_id=pid
    )
    print("✅ Watsonx model initialized.")
    return model
