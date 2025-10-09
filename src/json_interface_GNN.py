from gnn_drive_inference import json_processing
import json

# just follow this sample
test_json = json.dumps({
    "step": 710,
    "orig_id": "88640",
    "dest_id": "60",
    "amount": 1000.0,
    "orig_old_balance": 10000.0,
    "orig_new_balance": 9000.0,
    "dest_old_balance": 4000.0,
    "dest_new_balance": 5000.0
})


result_df = json_processing(test_json)