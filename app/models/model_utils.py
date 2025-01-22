import pandas as pd


def load_data(filepath='../data/creditcard.csv'):
    """
    Load the dataset from a CSV file.
    """
    try:
        print("Loading data from:", filepath)  # Debugging print
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(
            f"Error: File not found at {filepath}. Ensure the dataset is placed correctly.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def explore_data(data):
    """
    Explore the dataset for insights.
    """
    if data is not None:
        print("First few rows of the dataset:")
        print(data.head())

        print("\nDataset information:")
        print(data.info())

        print("\nClass distribution (imbalanced data):")
        print(data['Class'].value_counts())
    else:
        print("Data is None. Cannot explore.")
