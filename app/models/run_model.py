import sys
import json
from joblib import load

# Load the trained model
model = load("best_model_RF.pkl")  # Ensure this path is correct

# Read input data from the command line
input_data = json.loads(sys.argv[1])

# Make prediction
prediction = model.predict([input_data])

# Print the prediction as JSON output
print(json.dumps({"prediction": prediction.tolist()}))
