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

# Check for duplicate 
duplicate_values = csv_data.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_values}")

# Check value counts 
categorical_columns = csv_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(csv_data[column].value_counts())
    print()