# 🏦 Pulse4 — Bank Risk AI Agent

An intelligent **AI-powered risk analysis and visualization dashboard** built with **Python** and **Streamlit**, integrating **IBM Watsonx AI** for explainable financial risk monitoring.

---

## 🌐 Overview

**Pulse4** is a prototype system designed for **intelligent banking risk analysis and fraud detection**.  
It integrates **Graph Neural Networks (GNNs)**, **AI reasoning agents**, and **interactive visualization tools** into one cohesive interface.

The system aims to:
- Help financial institutions **detect and interpret suspicious transaction patterns**
- Use **LLMs from IBM Watsonx** to generate natural-language risk explanations
- Demonstrate **AI-human collaboration** in financial risk management and decision support

---

## ⚙️ System Architecture

The project follows a **dual-path architecture**:

- **Path A — GNN Model**  
  A Graph Neural Network analyzes transaction relationships to predict fraud probabilities.  
  The trained model is exported as a `best_model.pth` checkpoint and served through a JSON-based inference interface.

- **Path B — AI Agent & Visualization**  
  The agent layer integrates **IBM Watsonx Granite-3-2-8b** for reasoning and explanation.  
  The dashboard visualizes transaction networks (via **NetworkX** and **PyVis**) and provides dynamic risk scoring, threshold adjustment, and simulation capabilities.

All data, including daily transaction logs, are managed through a **Google Drive–linked dynamic database**, enabling real-time data retrieval and updates.

---

## 🧠 Core Features

✅ End-to-end automated setup and environment configuration  
✅ Interactive `.env` creation for IBM Watsonx credentials  
✅ Real-time fraud probability inference through GNN  
✅ Dynamic visualization of transaction graphs and risk domains  
✅ Agent-based natural-language reasoning and explanations  
✅ Daily data integration and simulation updates (`daily_data/`)

---

## 🚀 Quick Start

### Prerequisites
- Python **3.9+**
- IBM Watsonx account with valid API access  
- Stable Internet connection (for model inference)

---

### 🧩 Setup and Launch

Run the setup script — no manual installation required:

```bash
bash setup.sh
