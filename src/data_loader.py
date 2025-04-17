import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, thousands=',')
    print("Data types:\n", data.dtypes)
    missing = data.isnull().sum()
    print("\nMissing values:\n", missing[missing > 0])
    duplicates = data.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    print("\nTotal rows:", data.shape[0])
    return data
