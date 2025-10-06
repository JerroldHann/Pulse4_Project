def analyze_risk(identifier: str):
    import random
    score = round(random.uniform(0.3, 0.95), 2)
    if score > 0.8:
        label = "⚠️ High Risk"
    elif score > 0.6:
        label = "⚡ Medium Risk"
    else:
        label = "✅ Low Risk"
    return f"Entity '{identifier}' has a risk score of {score}. Level: {label}"
