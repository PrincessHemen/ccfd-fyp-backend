import subprocess
import json
import io
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.model_utils import preprocess_input_data

router = APIRouter()

@router.post("/predict/")
async def predict_fraud(file: UploadFile = File(...)):
    """
    Upload a CSV file with transactions, and get fraud predictions.
    """
    try:
        # Read uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Preprocess the data
        df_processed = preprocess_input_data(df)

        # Convert processed data to JSON string
        input_json = df_processed.to_json(orient="records")

        # Get the absolute path to run_model.py
        script_path = os.path.join(os.path.dirname(__file__), "../models/run_model.py")

        # Call run_model.py using subprocess
        result = subprocess.run(
            ["python", script_path, input_json],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Model execution failed")

        # Parse model output
        prediction = json.loads(result.stdout)

        # Include transaction IDs if available
        response = {"predictions": prediction["prediction"]}
        if "trans_num" in df.columns:
            response["transaction_ids"] = df["trans_num"].tolist()

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
