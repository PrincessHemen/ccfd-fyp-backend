import sys
import json
import os
from joblib import load

# Get absolute path of the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_RF.pkl")

# Load the trained model
model = load(MODEL_PATH)  # Use the full path


# Read input data from the command line
input_data = json.loads(sys.argv[1])

# Make prediction
prediction = model.predict([input_data])

# Print the prediction as JSON output
print(json.dumps({"prediction": prediction.tolist()}))
