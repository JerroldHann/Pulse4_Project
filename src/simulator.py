import json
import pandas as pd
from src.risk_engine import calc_risk_score

DATA_PATH = "data/daily_transactions.csv"

def save_and_predict(json_str: str):
    """Parse JSON input, save to CSV, and calculate risk score."""
    try:
        # Parse JSON
        record = json.loads(json_str)
        df = pd.DataFrame([record])

        # Save or append to CSV
        try:
            old = pd.read_csv(DATA_PATH)
            df = pd.concat([old, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv(DATA_PATH, index=False)

        # Simulate a prediction (use fraud_prob field if exists)
        prob = record.get("fraud_prob", 0.5)
        result = calc_risk_score(prob)

        return {
            "status": "✅ Data saved successfully.",
            "input": record,
            "prediction": {
                "risk_score": result["score"],
                "risk_level": result["level"],
                "recommendation": result["recommendation"]
            }
        }
    except json.JSONDecodeError as e:
        return {"error": f"❌ Invalid JSON: {e}"}
    except Exception as e:
        return {"error": f"⚠️ Unexpected error: {e}"}
