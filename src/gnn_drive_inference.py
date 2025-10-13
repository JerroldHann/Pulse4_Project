import os, json
import numpy as np
import pandas as pd

# ==================== 本地数据读取 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_transaction_raw with feature_v2.0.csv")

def load_local_csv(path: str) -> pd.DataFrame:
    """从本地 CSV 文件读取历史交易数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 未找到文件: {path}")
    df = pd.read_csv(path)
    print(f"✅ 从本地 CSV 读取历史数据：{len(df)} 条记录")
    return df

# =============== 辅助函数 ==================
def get_time_period(hour: int) -> int:
    if hour < 6: return 0
    elif hour < 12: return 1
    elif hour < 18: return 2
    else: return 3

def compute_behavior_mode(tx_count, var_ratio, recent_tx24):
    if tx_count >= 20 and 0.2 < var_ratio < 2:
        return "active"
    elif tx_count < 5:
        return "low_freq"
    elif var_ratio >= 2 or recent_tx24 >= 5:
        return "bursty"
    else:
        return "normal"

# =============== 主函数 ==================
def update_features(test_json: str, history_df: pd.DataFrame) -> pd.DataFrame:
    loaded = json.loads(test_json)
    if isinstance(loaded, dict):
        test_df = pd.DataFrame([loaded])
    elif isinstance(loaded, list):
        test_df = pd.DataFrame(loaded)
    else:
        raise ValueError("输入 JSON 格式错误，应为单条 dict 或多条 list[dict]")

    results = []

    for _, row in test_df.iterrows():
        test = row.to_frame().T.rename(columns={
            "orig_old_balance": "oldbalanceOrg",
            "orig_new_balance": "newbalanceOrig",
            "dest_old_balance": "oldbalanceDest",
            "dest_new_balance": "destbalanceDest"
        })
        if "destbalanceDest" in test.columns:
            test.rename(columns={"destbalanceDest": "newbalanceDest"}, inplace=True)

        test["isFraud"] = 0
        test["hour"] = test["step"] % 24
        test["time_period"] = test["hour"].map(get_time_period)

        orig_id, dest_id = str(test["orig_id"].iloc[0]), str(test["dest_id"].iloc[0])
        orig_hist = history_df[history_df["orig_id"].astype(str) == orig_id]
        dest_hist = history_df[history_df["dest_id"].astype(str) == dest_id]

        # 发起者
        if not orig_hist.empty:
            orig_tx_count = len(orig_hist)
            orig_30d_mean = orig_hist["amount"].mean()
            orig_30d_var = orig_hist["amount"].var(ddof=0)
            orig_recent_24h = orig_hist[orig_hist["step"] > test["step"].iloc[0] - 24]
            orig_tx_24h = len(orig_recent_24h)
            orig_avg_amt_24h = orig_recent_24h["amount"].mean() if orig_tx_24h > 0 else orig_30d_mean
        else:
            orig_tx_count = orig_tx_24h = 1
            orig_30d_mean = test["amount"].iloc[0]
            orig_30d_var = 0
            orig_avg_amt_24h = test["amount"].iloc[0]

        orig_curr_volatility = abs(test["amount"].iloc[0] - orig_30d_mean) / (orig_30d_mean + 1e-6)
        orig_behavior_mode = compute_behavior_mode(orig_tx_count, orig_curr_volatility, orig_tx_24h)

        # 接收者
        if not dest_hist.empty:
            dest_tx_count = len(dest_hist)
            dest_30d_mean = dest_hist["amount"].mean()
            dest_30d_var = dest_hist["amount"].var(ddof=0)
            dest_recent_24h = dest_hist[dest_hist["step"] > test["step"].iloc[0] - 24]
            dest_tx_24h = len(dest_recent_24h)
            dest_avg_amt_24h = dest_recent_24h["amount"].mean() if dest_tx_24h > 0 else dest_30d_mean
        else:
            dest_tx_count = dest_tx_24h = 1
            dest_30d_mean = test["amount"].iloc[0]
            dest_30d_var = 0
            dest_avg_amt_24h = test["amount"].iloc[0]

        dest_curr_volatility = abs(test["amount"].iloc[0] - dest_30d_mean) / (dest_30d_mean + 1e-6)
        dest_behavior_mode = compute_behavior_mode(dest_tx_count, dest_curr_volatility, dest_tx_24h)

        # balance ratio
        test["orig_balance_ratio"] = abs(test["oldbalanceOrg"] - test["newbalanceOrig"]) / (abs(test["oldbalanceOrg"]) + 1e-6)
        test["dest_balance_ratio"] = abs(test["oldbalanceDest"] - test["newbalanceDest"]) / (abs(test["oldbalanceDest"]) + 1e-6)

        # 汇总（命名统一）
        test["orig_tx_24h"] = orig_tx_24h
        test["orig_tx_72h"] = orig_tx_24h
        test["orig_tx_168h"] = orig_tx_24h
        test["orig_avgamt_24h"] = orig_avg_amt_24h
        test["orig_30d_mean"] = orig_30d_mean
        test["orig_30d_var"] = orig_30d_var
        test["orig_curr_volatility"] = orig_curr_volatility
        test["orig_behavior_mode"] = orig_behavior_mode

        test["dest_tx_24h"] = dest_tx_24h
        test["dest_tx_72h"] = dest_tx_24h
        test["dest_tx_168h"] = dest_tx_24h
        test["dest_avgamt_24h"] = dest_avg_amt_24h
        test["dest_30d_mean"] = dest_30d_mean
        test["dest_30d_var"] = dest_30d_var
        test["dest_curr_volatility"] = dest_curr_volatility
        test["dest_behavior_mode"] = dest_behavior_mode

        results.append(test)

    enriched_df = pd.concat(results, ignore_index=True)
    print(f"✅ 特征更新完成，共 {len(enriched_df)} 条交易记录。")
    return enriched_df

# =================== json processing ===================
def json_processing(json_input: str):
    print("🚀 正在从本地 CSV 文件中读取历史数据 ...")
    history_df = load_local_csv(DATA_PATH)

    enriched = update_features(json_input, history_df)
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enriched_transactions.csv")
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ 已保存特征增强数据至: {OUTPUT_PATH}")

    # 自动调用推理脚本
    import subprocess
    print("🚀 正在执行模型推理 ...")
    subprocess.run(["python", "model_gnn.py"])
