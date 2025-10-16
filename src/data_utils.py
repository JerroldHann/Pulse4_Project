import os
import pandas as pd
from datetime import datetime, timedelta

DAILY_FOLDER = "daily_data"

def _resolve_folder(folder: str = DAILY_FOLDER) -> str:
    """Resolve daily_data path robustly whether under project root or src/."""
    here = os.path.dirname(__file__)
    p1 = os.path.join(here, folder)
    p2 = os.path.join(os.path.dirname(here), folder)
    if os.path.exists(p1) or not os.path.exists(p2):
        os.makedirs(p1, exist_ok=True)
        return p1
    os.makedirs(p2, exist_ok=True)
    return p2

def split_dataset_by_day(source_csv="test_predictions_v2.0.csv", output_folder=DAILY_FOLDER):
    """
    Split a dataset (with 'step' column in hours) into daily files:
    daily_transactions_YYYYMMDD.csv
    """
    folder = _resolve_folder(output_folder)
    df = pd.read_csv(source_csv)
    if "step" not in df.columns:
        raise ValueError("source CSV must contain 'step' column (0..743 for 31 days).")
    df["day"] = (df["step"] // 24).astype(int)
    today = datetime.now().date()
    start_date = today - timedelta(days=int(df["day"].max()))
    for d in sorted(df["day"].unique()):
        date_str = (start_date + timedelta(days=int(d))).strftime("%Y%m%d")
        df[df["day"] == d].to_csv(os.path.join(folder, f"daily_transactions_{date_str}.csv"), index=False)

def resolve_today_csv(output_folder=DAILY_FOLDER) -> str:
    """Return today's CSV; if missing, fall back to latest available file; if none, raise."""
    folder = _resolve_folder(output_folder)
    today_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(folder, f"daily_transactions_{today_str}.csv")
    if os.path.exists(path):
        return path
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    if files:
        return os.path.join(folder, files[-1])
    raise FileNotFoundError("No daily CSV available. You can run split_dataset_by_day(...) first.")

def load_data_by_days_ago(days_ago=0, folder=DAILY_FOLDER) -> pd.DataFrame:
    """Load CSV for N days ago."""
    folder_path = _resolve_folder(folder)
    target = (datetime.now().date() - timedelta(days=int(days_ago))).strftime("%Y%m%d")
    csv_path = os.path.join(folder_path, f"daily_transactions_{target}.csv")
    if not os.path.exists(csv_path):
        available = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        raise FileNotFoundError(f"Missing {csv_path}. Available: {available[:10]} ...")
    return pd.read_csv(csv_path)

def build_feature_store(csv_path: str) -> dict:
    """
    Map transaction_id -> fraud_prob_pred (float).
    Compatible with test_predictions_v2.0 structure.
    """
    df = pd.read_csv(csv_path)
    need = ["transaction_id", "fraud_prob_pred"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return dict(zip(df["transaction_id"].astype(str), df["fraud_prob_pred"].astype(float)))

