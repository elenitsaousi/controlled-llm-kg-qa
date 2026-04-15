import json

with open('data/infineon/infineon_dataset_30.json') as f:
    data = json.load(f)

# Βρες τα missing types
missing_types = [
    "tier1_future_regional",
    "semi_future_regional", 
    "oem_future_regional",
    "semi_order_cancel",
    "tier1_current_demand_auto",
    "oem_current_demand"
]

for item in data:
    if item['id'] in missing_types:
        print(f"\n=== {item['id']} ===")
        print(f"Q: {item['question']}")
        print(f"A: {item['query']}")