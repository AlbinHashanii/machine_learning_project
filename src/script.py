import pandas as pd

csv_data = pd.read_csv("synthetic_fraud_dataset.csv")

# Printing the data types
print("Data types for each field:")
print(csv_data.dtypes)
print()

# Check for missing values
missing_values = csv_data.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing values in columns:")
print(missing_values)
print()