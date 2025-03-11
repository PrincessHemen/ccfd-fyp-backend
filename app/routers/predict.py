import pandas as pd
import joblib
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.model_utils import preprocess_data, MODEL_PATH

router = APIRouter()

# Load the trained model
model = joblib.load(MODEL_PATH)


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
        df_processed = preprocess_data(df)

        # Predict fraud cases
        predictions = model.predict(df_processed)

        # Return predictions along with original transaction identifiers if available
        result = {"predictions": predictions.tolist()}

        # If there's a transaction ID column in the original data, include it in the response
        if "trans_num" in df.columns:
            result["transaction_ids"] = df["trans_num"].tolist()

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
