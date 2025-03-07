import pandas as pd

csv_data = pd.read_csv("synthetic_fraud_dataset.csv")

print("Data types for each field:")
print(csv_data.dtypes)
print()
