import json
import os
import pandas as pd
from datetime import datetime
from .data_utils import resolve_today_csv, _resolve_folder
from gnn_drive_inference import json_processing

REQUIRED_INPUT_FIELDS = [
    "step","transaction_id","orig_id","dest_id","amount",
    "orig_balance_ratio","dest_balance_ratio",
    "orig_curr_volatility","dest_curr_volatility"
]

def _fake_predict(row: dict) -> dict:
    """
    Dumb heuristic just to show the flow.
    """
    amt = float(row.get("amount", 0.0))
    vol = float(row.get("orig_curr_volatility", 0.05)) + float(row.get("dest_curr_volatility", 0.05))
    ratio = float(row.get("orig_balance_ratio", 0.5)) + float(row.get("dest_balance_ratio", 0.5))
    prob = max(0.01, min(0.99, 0.2 + 0.00005 * amt + 0.2 * vol + 0.1 * ratio))
    is_fraud = int(prob > 0.7)
    return {
        "transaction_id": row.get("transaction_id"),
        "fraud_prob_pred": round(prob, 4),
        "isFraud_pred": is_fraud,
        "risk_score": round(600 + 20 / 0.693147 * ( (1-prob)/max(prob,1e-12) / 50 )**0.0, 1)  # just a placeholder
    }

def save_and_predict(user_json_str: str) -> dict:
    """
    Parse JSON, append to today's daily CSV (create if not exists),
    return fake prediction as JSON.
    """
    try:
        data = json.loads(user_json_str)
        if not isinstance(data, dict):
            return {"error": "Input must be a single JSON object."}
        for f in REQUIRED_INPUT_FIELDS:
            if f not in data:
                return {"error": f"Missing field: {f}"}

        pred = json_processing(data)

        # ensure today's CSV exists or create
        folder = _resolve_folder()
        today_str = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(folder, f"daily_transactions_{today_str}.csv")

        row = {
            **data,
            "fraud_prob_pred": pred["fraud_prob_pred"],
            "isFraud_pred": pred["isFraud_pred"],
        }

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(csv_path, index=False)

        return {
            "status": f"Saved to {os.path.basename(csv_path)}",
            "prediction": pred
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format."}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
