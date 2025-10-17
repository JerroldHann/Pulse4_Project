import pandas as pd

def get_transactions(client_name: str = "", min_prob: float = 0.5, start_step: int = None, end_step: int = None, probability_threshold:float = None) -> pd.DataFrame:
    """
    Return filtered transactions, re-reading the CSV each call (so Tab3 always fresh).
    This function will only read from the specific file: 'data/test_predictions_v2.0.csv'.
    """
    # 读取数据
    file_path = "/home/yjing/Pulse4_Project/data/test_predictions_v2.0.csv"
    df = pd.read_csv(file_path)

    if df.empty:
        return df

    # 选择需要的列
    cols = [c for c in ["transaction_id", "orig_id", "dest_id", "amount", "fraud_prob_pred", "isFraud_pred", "step"] if c in df.columns]
    df = df[cols].copy()

    # 根据客户名过滤数据
    if client_name:
        s = str(client_name)
        df = df[(df["orig_id"].astype(str) == s) | (df["dest_id"].astype(str) == s)]

    # 根据欺诈概率过滤
    if "fraud_prob_pred" in df.columns:
        df = df[df["fraud_prob_pred"].astype(float) >= float(min_prob)]

    # 根据 step 过滤数据（如果提供了 start_step 和 end_step）
    if start_step is not None and end_step is not None:
        df = df[(df["step"] >= start_step) & (df["step"] <= end_step)]
    
    # 按照欺诈概率排序
    return df.sort_values(by="fraud_prob_pred", ascending=False, na_position="last")
