import pandas as pd
from src.data_utils import load_feature_store

def get_transactions(client_name="", min_prob=0.0):
    store = load_feature_store()
    df = pd.DataFrame(store).T.reset_index(names=["tx_id"])
    if client_name:
        df = df[df["orig_account"].str.contains(client_name, na=False) |
                df["dest_account"].str.contains(client_name, na=False)]
    if min_prob:
        df = df[df["fraud_prob"] >= min_prob]
    return df
