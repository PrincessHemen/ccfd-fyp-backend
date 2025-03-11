import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Define paths - adjusted to ensure we get the correct path to models directory
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # Gets the app directory
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Points to app/models
# Points to the exact model file
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_RF.pkl")


# Function to preprocess user-uploaded CSV


def preprocess_input_data(file_path):
    """
    Preprocesses a user-uploaded CSV file to match the format required for model prediction.

    Args:
        file_path (str): Path to the uploaded CSV file.

    Returns:
        pd.DataFrame: Processed dataframe ready for model prediction.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert datetime columns
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

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
