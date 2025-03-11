import pandas as pd

csv_data = pd.read_csv("atk-investimet-tvsh.csv")

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

# Advanced missing values check
def adv_miss_check(df):
        no_info_types = [" ", "-", "--", "na", "n/a", "?", "no info", "missing info", "*"]
        df = df.map(lambda x: str(x).lower() if isinstance(x, str) else x)
        missing_rows = df[df.apply(lambda row: row.isin(no_info_types).any(), axis=1)]
        return missing_rows
adv_miss_check(csv_data)

# Check for duplicate
duplicate_values = csv_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_values}")

# Check value counts
categorical_columns = csv_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(csv_data[column].value_counts())
    print()
