import json
import os
import pandas as pd
from datetime import datetime
from .data_utils import resolve_today_csv, _resolve_folder
from .gnn_drive_inference import json_processing
import streamlit as st

REQUIRED_INPUT_FIELDS = [
    "step","orig_id","dest_id","amount",
    "orig_old_balance","orig_new_balance",
    "dest_old_balance","dest_new_balance"
]



def save_and_predict(user_json_str: str):
    
    try:
        data = json.loads(user_json_str)
      

        pred = json_processing(data)
        print(pred)
        st.write("Prediction Status:")
        st.json(pred)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format."}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}