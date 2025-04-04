import sys
import json
import os
import pandas as pd
from joblib import load

# Absolute paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "collab_rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.pkl")

# Load model, scaler, and selected features
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
selected_features = load(FEATURES_PATH)

# Read input data from command line
input_data = json.loads(sys.argv[1])

# Handle either a single dictionary or a list of dictionaries
if isinstance(input_data, list):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(input_data)
else:
    # Convert single dictionary to DataFrame
    df = pd.DataFrame([input_data])

# Ensure all required features are present
for feature in selected_features:
    if feature not in df.columns:
        df[feature] = 0  # Add missing features with default value

# Filter and reorder columns
df = df[selected_features]

# Scale input
scaled_input = scaler.transform(df)

# Predict
prediction = model.predict(scaled_input)

# Print prediction
print(json.dumps({"prediction": prediction.tolist()}))
