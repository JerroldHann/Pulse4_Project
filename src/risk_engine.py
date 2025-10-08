import numpy as np
from src.data_utils import load_feature_store

def calc_risk_score(prob):
    """
    Convert fraud probability into a standardized risk score,
    along with risk level and recommended action.
    """
    odds = (1 - prob) / prob
    score = 600 + 20 / np.log(2) * np.log(odds / 50)

    if score >= 700:
        level, rec = "Low Risk", "Normal transaction"
    elif score >= 600:
        level, rec = "Moderate Risk", "Monitor recommended"
    elif score >= 500:
        level, rec = "High Risk", "Manual review required"
    else:
        level, rec = "Severe Risk", "Immediate investigation needed"

    return {
        "score": round(score, 1),
        "level": level,
        "recommendation": rec
    }


def analyze_risk(identifier: str):
    """
    Analyze the risk level of a given transaction or account.
    """
    store = load_feature_store()
    tx = store.get(identifier)

    if not tx:
        return f"No transaction found for {identifier}."

    result = calc_risk_score(tx["fraud_prob"])
    return (
        f"{identifier}: Score = {result['score']} "
        f"({result['level']}), Recommendation: {result['recommendation']}"
    )
