import os
import numpy as np
import pandas as pd

def create_sample_data(filename="data/daily_transactions.csv", num_records=100):
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    data = []
    accounts = [f"A{i:03d}" for i in range(1, 21)]
    for i in range(1, num_records + 1):
        tid = f"T{i:03d}"
        a, b = np.random.choice(accounts, 2, replace=False)
        amount = np.random.uniform(10, 10000)
        fraud_prob = np.random.uniform(0.01, 0.9)
        is_fraud = 1 if fraud_prob > 0.75 else 0
        data.append({
            "transaction_id": tid,
            "orig_account": a,
            "dest_account": b,
            "amount": round(amount, 2),
            "fraud_prob": round(fraud_prob, 4),
            "is_fraud": is_fraud
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Created {filename}")
    return df

def load_feature_store(path="data/daily_transactions.csv"):
    if not os.path.exists(path):
        create_sample_data(path)
    df = pd.read_csv(path)
    store = {r["transaction_id"]: dict(r) for _, r in df.iterrows()}
    return store
