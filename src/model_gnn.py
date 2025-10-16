# ==================== inference_from_saved_model.py ====================
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json, pickle, os
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==================== 1️⃣ 自动定位目录 ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # src/
BASE_DIR = os.path.dirname(CURRENT_DIR)                         # 项目根目录
MODEL_DIR = os.path.join(BASE_DIR, "model")                     # model 文件夹

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scalers.pkl")
MAP_PATH = os.path.join(MODEL_DIR, "mapping.pkl")

print(f"📂 当前工作目录: {CURRENT_DIR}")
print(f"📦 模型目录: {MODEL_DIR}")

# ==================== 2️⃣ 加载配置、映射与模型参数 ====================
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
with open(SCALER_PATH, "rb") as f:
    scalers = pickle.load(f)
with open(MAP_PATH, "rb") as f:
    mapping = pickle.load(f)

node_scaler = scalers["node_scaler"]
edge_scaler = scalers["edge_scaler"]
node2idx = mapping["node2idx"]
unique_nodes = mapping["unique_nodes"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 配置加载完成，使用设备：{DEVICE}")

# ==================== 3️⃣ 数据加载 ====================
CSV_PATH = os.path.join(CURRENT_DIR, "enriched_transactions.csv")

if not os.path.exists(CSV_PATH):
    # 若当前目录无该文件，则尝试项目根目录（防止在 src/ 与根目录之间切换）
    alt_path = os.path.join(BASE_DIR, "enriched_transactions.csv")
    if os.path.exists(alt_path):
        CSV_PATH = alt_path
    else:
        raise FileNotFoundError(
            f"❌ 未找到特征文件 enriched_transactions.csv\n"
            f"请确认该文件是否在以下路径之一：\n{CURRENT_DIR}\n{BASE_DIR}"
        )

print(f"📄 使用特征增强数据集：{CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ 成功读取数据：{len(df)} 条记录")



# ==================== 4️⃣ 构建图结构 ====================
src = df['orig_id'].map(node2idx).to_numpy(dtype=np.int64)
dst = df['dest_id'].map(node2idx).to_numpy(dtype=np.int64)
edge_index = np.vstack([src, dst])
edge_y = df['isFraud'].astype(int).to_numpy()

edge_attr = df[['amount', 'time_period']].to_numpy(dtype=np.float64)
edge_risk_weight = 0.2 + 0.8 * df['isFraud'].astype(float).to_numpy()
edge_attr = np.hstack([edge_attr, edge_risk_weight.reshape(-1, 1)])
edge_attr = np.nan_to_num(edge_attr)
edge_attr = edge_scaler.transform(edge_attr).astype(np.float32)

# === 节点特征构造 ===
grp_orig = df.groupby('orig_id')
grp_dest = df.groupby('dest_id')

def safe_stat(g, col, func='mean', fill=0):
    if func == 'mean':
        return g[col].mean().reindex(unique_nodes, fill_value=fill).to_numpy()
    elif func == 'sum':
        return g[col].sum().reindex(unique_nodes, fill_value=fill).to_numpy()
    elif func == 'count':
        return g.size().reindex(unique_nodes, fill_value=fill).to_numpy()

# 基础统计特征
orig_count = safe_stat(grp_orig, 'amount', 'count')
orig_sum = safe_stat(grp_orig, 'amount', 'sum')
orig_mean = safe_stat(grp_orig, 'amount', 'mean')
dest_count = safe_stat(grp_dest, 'amount', 'count')
dest_sum = safe_stat(grp_dest, 'amount', 'sum')
dest_mean = safe_stat(grp_dest, 'amount', 'mean')

# balance ratio & volatility
orig_bal = safe_stat(grp_orig, 'orig_balance_ratio')
dest_bal = safe_stat(grp_dest, 'dest_balance_ratio')
orig_vol = safe_stat(grp_orig, 'orig_curr_volatility')
dest_vol = safe_stat(grp_dest, 'dest_curr_volatility')

# 30天统计
for col in ['orig_30d_mean','orig_30d_var','dest_30d_mean','dest_30d_var']:
    if col not in df.columns:
        df[col] = 0.0

orig_30d_mean = df.groupby('orig_id')['orig_30d_mean'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
dest_30d_mean = df.groupby('dest_id')['dest_30d_mean'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
orig_30d_var  = df.groupby('orig_id')['orig_30d_var'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
dest_30d_var  = df.groupby('dest_id')['dest_30d_var'].mean().reindex(unique_nodes, fill_value=0).to_numpy()

# 时间窗口统计
if {'orig_tx_24h','orig_tx_72h','orig_tx_168h','orig_avg_amt_24h'}.issubset(df.columns):
    freq_24h = df.groupby('orig_id')['orig_tx_24h'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
    freq_72h = df.groupby('orig_id')['orig_tx_72h'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
    freq_168h = df.groupby('orig_id')['orig_tx_168h'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
    avg_24h = df.groupby('orig_id')['orig_avg_amt_24h'].mean().reindex(unique_nodes, fill_value=0).to_numpy()
else:
    freq_24h = freq_72h = freq_168h = avg_24h = np.zeros(len(unique_nodes))

# 行为模式 One-hot
modes = ['active','normal','low_freq','bursty']
mode_map = {m:i for i,m in enumerate(modes)}
if 'orig_behavior_mode' in df.columns:
    orig_mode_idx = df.groupby('orig_id')['orig_behavior_mode'].first().reindex(unique_nodes, fill_value='normal').map(mode_map).to_numpy()
else:
    orig_mode_idx = np.zeros(len(unique_nodes), dtype=int)
orig_mode_oh = np.eye(len(modes))[orig_mode_idx]

node_feats = np.vstack([
    orig_count, orig_sum, orig_mean,
    dest_count, dest_sum, dest_mean,
    orig_bal, dest_bal, orig_vol, dest_vol,
    orig_30d_mean, orig_30d_var, dest_30d_mean, dest_30d_var,
    freq_24h, freq_72h, freq_168h, avg_24h
]).T.astype(np.float64)
node_feats = np.hstack([node_feats, orig_mode_oh.astype(np.float64)])

node_feats = np.nan_to_num(node_feats)
node_feats = node_scaler.transform(node_feats).astype(np.float32)

data = Data(
    x=torch.tensor(node_feats, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    y=torch.tensor(edge_y, dtype=torch.float)
).to(DEVICE)

# ==================== 5️⃣ 模型定义与加载 ====================
class EdgeSAGE(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(node_in, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_in, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        src, dst = edge_index
        src_emb, dst_emb = x[src], x[dst]
        edge_feat = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        return self.edge_mlp(edge_feat).view(-1)

model = EdgeSAGE(node_in=data.num_node_features, edge_in=edge_attr.shape[1], hidden_dim=config["EMBED_DIM"]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ 模型权重加载成功")

# ==================== 6️⃣ 推理 ====================
with torch.no_grad():
    logits = model(data.x, data.edge_index, data.edge_attr)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)

# === 输出结果 ===
df_out = pd.DataFrame({
    "step": df["step"],
    "orig_id": df["orig_id"],
    "dest_id": df["dest_id"],
    "amount": df["amount"],
    "fraud_prob_pred": probs,
    "isFraud_pred": preds,
    "orig_behavior_mode": df.get("orig_behavior_mode", "normal"),
    "dest_behavior_mode": df.get("dest_behavior_mode", "normal")
})

df_out = df_out.sort_values("step").reset_index(drop=True)
df_out["transaction_id"] = [f"1{int(row.step):03d}{i:05d}" for i, row in enumerate(df_out.itertuples(), start=1)]

OUTPUT_PATH = os.path.join(CURRENT_DIR, "inference_result.csv")
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"\n📄 推理完成，结果已保存至：{OUTPUT_PATH}\n")
print("📊 推理结果预览：")
string_out = df_out.to_string(index=False)
print(string_out)
st.json(string_out)

