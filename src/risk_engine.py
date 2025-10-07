import numpy as np
from src.data_utils import load_feature_store

def calc_risk_score(prob):
    odds = (1 - prob) / prob
    score = 600 + 20 / np.log(2) * np.log(odds / 50)
    if score >= 700: lvl, rec = "低风险", "正常交易"
    elif score >= 600: lvl, rec = "中等风险", "建议监控"
    elif score >= 500: lvl, rec = "中高风险", "人工复核"
    else: lvl, rec = "高风险", "立即调查"
    return {"score": round(score,1), "level": lvl, "recommendation": rec}

def analyze_risk(identifier: str):
    store = load_feature_store()
    tx = store.get(identifier)
    if not tx:
        return f"未找到交易 {identifier}"
    res = calc_risk_score(tx['fraud_prob'])
    return f"{identifier}: 分数={res['score']} ({res['level']})，建议：{res['recommendation']}"
