import subprocess
import json
import io
import os
import pandas as pd
import sys
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

        # Convert processed data to list of dictionaries
        records = df_processed.to_dict(orient="records")

        # Get the absolute path to run_model.py
        script_path = os.path.join(os.path.dirname(
            __file__), "../models/run_model.py")

        # Process each transaction separately to avoid memory issues with large files
        all_predictions = []

        for record in records:
            # Convert single record to JSON string
            input_json = json.dumps(record)

            # Call run_model.py using subprocess
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, script_path, input_json],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model execution failed: {result.stderr}"
                )

            # Parse model output
            prediction_result = json.loads(result.stdout)
            all_predictions.extend(prediction_result["prediction"])

        # Include transaction IDs if available
        response = {"predictions": all_predictions}
        if "trans_num" in df.columns:
            response["transaction_ids"] = df["trans_num"].tolist()

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
