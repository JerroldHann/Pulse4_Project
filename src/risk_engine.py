import numpy as np
from .data_utils import resolve_today_csv, build_feature_store
import pandas as pd

def calc_risk_score(prob: float):
    """Classic scorecard transform with base score 600 and PDO=20."""
    prob = float(prob)
    odds = (1 - prob) / max(prob, 1e-12)
    score = 600 + (20 / np.log(2)) * np.log(odds / 50)
    if score >= 700:
        lvl, rec = "Low", "No action"
    elif score >= 600:
        lvl, rec = "Medium", "Monitor"
    elif score >= 500:
        lvl, rec = "Medium-High", "Manual review"
    else:
        lvl, rec = "High", "Investigate immediately"
    return {"score": round(score, 1), "level": lvl, "recommendation": rec}

def analyze_risk(identifier: str) -> str:
    """
    If identifier matches a transaction_id in today's CSV -> score it.
    Otherwise try to find first tx involving that account and score that tx.
    """
    csv_path = resolve_today_csv()
    store = build_feature_store(csv_path)
    df = pd.read_csv(csv_path)

    # 1) direct tx id
    if identifier in store:
        prob = store[identifier]
        res = calc_risk_score(prob)
        return f"{identifier}: score={res['score']} ({res['level']}), recommendation: {res['recommendation']}"

    # 2) first transaction for this account
    id_str = str(identifier)
    mask = (df["orig_id"].astype(str) == id_str) | (df["dest_id"].astype(str) == id_str)
    if mask.any():
        row = df[mask].iloc[0]
        prob = float(row.get("fraud_prob_pred", 0.01))
        res = calc_risk_score(prob)
        txid = row["transaction_id"]
        return f"{id_str} via {txid}: score={res['score']} ({res['level']}), recommendation: {res['recommendation']}"
    return f"Not found: {identifier}"
