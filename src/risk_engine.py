import numpy as np
from .data_utils import resolve_today_csv, build_feature_store
import pandas as pd
import os

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
import os
import pandas as pd
import numpy as np

def composite_risk_index(prob, amount, transaction_id=None, folder="data", 
                         sigma1=0.6, sigma2=0.3, sigma3=0.1, cap_percentile=95, 
                         verbose=True):
    """
    Composite Risk Index (RI)
    ------------------------------------------------------
    RI = σ1 * P(fraud) + σ2 * Â + σ3 * ln(P/(1-P))
    ------------------------------------------------------
    Features:
        - Amount baseline A₀ is calculated from a single file, data/test_predictions_v2.0.csv (full history P95)
        - A₀ is cached, and subsequent calls use the cached value to avoid repeated file reading
    ------------------------------------------------------
    Parameters:
        prob: Fraud probability (list or array)
        amount: Transaction amount (list or array)
        folder: Path to the data folder
        σ1, σ2, σ3: Weight coefficients (tunable)
        cap_percentile: Cap percentile for A₀ (default P95)
    Outputs:
        dict containing:
            - RI: Composite Risk Index
            - risk_level: Risk level
            - explanation: Risk explanation
            - recommendation: Recommended actions
    """

    # ---------- Step 1. Calculate or read the cached global A₀ ----------
    try:
        # ✅ Use cached value if already calculated
        if hasattr(composite_risk_index, "_A0_cache"):
            A0 = composite_risk_index._A0_cache
            if verbose:
                print(f"📊 Loaded cached global amount percentile A₀ (P{cap_percentile}) = {A0:.2f}")
        else:
            # Use correct path for the file
            file_path = os.path.join(os.path.dirname(__file__), "/home/yjing/Pulse4_Project/data/test_predictions_v2.0.csv")
            df = pd.read_csv(file_path, usecols=["amount"])
            all_amounts = df["amount"].dropna().values

            if len(all_amounts) > 0:
                A0 = np.nanpercentile(all_amounts, cap_percentile)
                # ✅ Cache the calculated result
                composite_risk_index._A0_cache = A0
            else:
                A0 = 1.0

            if verbose:
                print(f"📊 Global amount percentile A₀ (P{cap_percentile}) = {A0:.2f}, based on {len(all_amounts):,} transactions.")

    except Exception as e:
        print(f"⚠️ Failed to calculate global A₀, using default A₀=1.0. Error: {e}")
        A0 = 1.0

    # ---------- Step 2. Normalize the amount ----------
    amount = np.array(amount, dtype=float)
    A_tilde = np.clip(amount / (A0 + 1e-9), 0, 1)

    # ---------- Step 3. Fraud probability and logit term ----------
    prob = np.array(prob, dtype=float)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    logit_term = np.log(prob / (1 - prob))

    # ---------- Step 4. Composite Risk Index Calculation ----------
    RI = sigma1 * prob + sigma2 * A_tilde + sigma3 * logit_term

    # ---------- Step 5. Risk level classification ----------
    levels, explanations, recommendations = [], [], []
    for r in RI:
        if r >= 0.9:
            levels.append("High Risk")
            explanations.append("Both fraud probability and transaction amount are extremely high, likely indicating fraudulent transactions.")
            recommendations.append("Immediately freeze the account or block the transaction, and escalate for further review.")
        elif r >= 0.6:
            levels.append("Suspicious Risk")
            explanations.append("Fraud probability or amount is relatively high, indicating potential risk.")
            recommendations.append("Recommend manual review or trigger secondary verification.")
        elif r >= 0.3:
            levels.append("Normal Risk")
            explanations.append("The transaction risk is at an average level.")
            recommendations.append("Proceed as usual, but continuous monitoring is advised.")
        else:
            levels.append("Low Risk")
            explanations.append("Both the transaction amount and the model output are relatively low.")
            recommendations.append("No intervention necessary.")

    # ---------- Step 6. Return results ----------
    result = {
        "transaction_id": transaction_id,
        "input_prob": prob.tolist(),
        "amount": amount.tolist(),
        "A_tilde": A_tilde.tolist(),
        "A0_global": A0,
        "RI": RI.tolist(),
        "risk_level": levels,
        "explanation": explanations,
        "recommendation": recommendations
    }

    if verbose:
        print("\n🎯 Composite Risk Index Calculation Results:")
        for i in range(len(prob)):
            print(f"📌 Transaction {transaction_id or '-'}")
            print(f"   Fraud Probability: {prob[i]:.4f}")
            print(f"   Normalized Amount: {A_tilde[i]:.3f}  (Global A₀={A0:.2f})")
            print(f"   Composite Risk Index (RI): {RI[i]:.3f}")
            print(f"   Risk Level: {levels[i]}")
            print(f"   Recommendation: {recommendations[i]}\n")

    return result
