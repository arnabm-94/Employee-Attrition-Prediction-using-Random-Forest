import pandas as pd
import joblib

# 1. Load model and scaler
model = joblib.load("final_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Load test data
df = pd.read_csv("test_data_RF.csv")

# 3. Scale numerical values (model requires scaled input)
X = scaler.transform(df)

# 4. Predict attrition (1 = leave, 0 = stay)
predictions = model.predict(X)

# 5. Show results
for i, pred in enumerate(predictions):
    print(f"Employee {i+1} attrition prediction: {'YES (Will Leave)' if pred==1 else 'NO (Will Stay)'}")
