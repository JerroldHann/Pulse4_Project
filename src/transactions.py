import pandas as pd
from .data_utils import resolve_today_csv, load_data_by_days_ago

def get_transactions(client_name: str = "", min_prob: float = 0.5,
                     start_days_ago: int | None = None,
                     end_days_ago:   int | None = None) -> pd.DataFrame:
    """
    Return filtered transactions, re-reading CSV each call (so Tab3 always fresh).
    If a range is provided, concat those days; else use today's CSV.
    """
    if start_days_ago is not None and end_days_ago is not None:
        frames = []
        for d in range(int(start_days_ago), int(end_days_ago) - 1, -1):
            try:
                frames.append(load_data_by_days_ago(d))
            except FileNotFoundError:
                continue
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = pd.read_csv(resolve_today_csv())

    if df.empty:
        return df

    cols = [c for c in ["transaction_id","orig_id","dest_id","amount","fraud_prob_pred","isFraud_pred","step"] if c in df.columns]
    df = df[cols].copy()

    if client_name:
        s = str(client_name)
        df = df[(df["orig_id"].astype(str) == s) | (df["dest_id"].astype(str) == s)]

    if "fraud_prob_pred" in df.columns:
        df = df[df["fraud_prob_pred"].astype(float) >= float(min_prob)]

    return df.sort_values(by="fraud_prob_pred", ascending=False, na_position="last")
