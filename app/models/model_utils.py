import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_RF.pkl")

# uvicorn app.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs

# Function to preprocess user-uploaded CSV


def preprocess_input_data(df):
    """
    Preprocesses a user-uploaded CSV file to match the format required for model prediction.

    Args:
        df (pd.DataFrame): Raw DataFrame from uploaded CSV.

    Returns:
        pd.DataFrame: Processed DataFrame ready for model prediction.
    """
    # Convert datetime columns
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"], dayfirst=True)

    # Extract age and transaction hour
    df["age"] = df["dob"].apply(lambda x: datetime.now().year - x.year)
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour

    # Normalize transaction amount using MinMaxScaler (fitting on current data)
    scaler = MinMaxScaler()
    df["amt_scaled"] = scaler.fit_transform(df[["amt"]])

    # One-hot encode categorical variables
    categorical_columns = ["category", "gender", "state", "job"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Drop unnecessary columns
    columns_to_drop = ["cc_num", "first", "last", "street", "city", "merchant",
                       "trans_date_trans_time", "dob", "trans_num"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Ensure "is_fraud" column is not included (since this is for prediction)
    if "is_fraud" in df.columns:
        df.drop(columns=["is_fraud"], inplace=True)

    return df
