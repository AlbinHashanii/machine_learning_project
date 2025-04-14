import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, zscore
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier

csv_data = pd.read_csv("atk-investimet-tvsh.csv", thousands=',')

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
print()

#Count the total number of rows
total_rows = csv_data.shape[0]
print("Total number of rows:", total_rows)
print()

numerical_cols = [
    "Viti", "Muaji", "Tatimpaguesit",
    "Blerjet dhe importet investive pa TVSH",
    "Blerjet dhe importet investive me TVSH jo te zbritshme",
    "Importet investive me norme 18%",
    "Importet investive me norme 8%",
    "Blerjet investive vendore me norme 18%",
    "Blerjet investive vendore me norme 8%"
]

# Aggregate numerical data
pd.set_option('display.max_columns', None)
aggregated_data = csv_data[numerical_cols].agg(['mean', 'median', 'sum', 'std', 'count'])
print("Aggregated Data:")
print(aggregated_data)
print()

# Sample 10% of the dataset randomly
csv_data_sampled = csv_data.sample(frac=0.1)
print(f"Sampled Data Shape: {csv_data_sampled.shape}")
print()

# Check value counts
categorical_columns = csv_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(csv_data[column].value_counts())
    print()

# Categorical columns
categorical_columns = ['Pershkrimi', 'Statusi', 'Komuna']

label_encoders = {}
for col in categorical_columns:
    if col in csv_data.columns:
        le = LabelEncoder()
        csv_data[col] = le.fit_transform(csv_data[col])
        label_encoders[col] = le
    else:
        print(f"Column '{col}' not found in the dataset. Please verify the column names.")



# Calculate skewness
skewness_values = csv_data[numerical_cols].apply(skew, nan_policy='omit')

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(csv_data[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(csv_data[col].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(csv_data[col].median(), color='orange', linestyle='-', label='Median')
    plt.title(f"Distribution of {col} with Skewness: {skew(csv_data[col], nan_policy='omit'):.2f}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

print("Skewness of Numerical Columns:")
print(skewness_values)

# Apply Z-score method for outlier removal
z_scores = csv_data[numerical_cols].apply(zscore, nan_policy='omit')
threshold_z = 2

outlier_indices = np.where(np.abs(z_scores) > threshold_z)[0]

csv_data_no_outliers = csv_data.drop(outlier_indices)

csv_data_no_outliers.to_csv("Investimet-2024-cleaned-no-outliers.csv", index=False)
print()

# Visualization without Outliers
print("Visualizing distributions without outliers:")
skewness_values_no_outliers = csv_data_no_outliers[numerical_cols].apply(skew, nan_policy='omit')

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(csv_data_no_outliers[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(csv_data_no_outliers[col].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(csv_data_no_outliers[col].median(), color='orange', linestyle='-', label='Median')
    plt.title(f"Distribution of {col} without Outliers\nSkewness: {skew(csv_data_no_outliers[col], nan_policy='omit'):.2f}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
print("Skewness of Numerical Columns without Outliers:")
print(skewness_values_no_outliers)
print()

target_column = 'Statusi'
if target_column not in csv_data_no_outliers.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

X = csv_data_no_outliers.drop(columns=[target_column])
y = csv_data_no_outliers[target_column]

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

target_encoder = label_encoders[target_column]
plt.figure(figsize=(12, 6))

# Before SMOTE
plt.subplot(1, 2, 1)
counts_before = y_train.value_counts().sort_index()
original_labels_before = target_encoder.inverse_transform(counts_before.index)
plt.bar(original_labels_before, counts_before.values)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Target Classes")
plt.ylabel("Count")
plt.xticks(rotation=90)

# After SMOTE
plt.subplot(1, 2, 2)
counts_after = y_train_res.value_counts().sort_index()
original_labels_after = target_encoder.inverse_transform(counts_after.index)
plt.bar(original_labels_after, counts_after.values)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Target Classes")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

nb_classifier.fit(X_train_res, y_train_res)

y_pred_nb = nb_classifier.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
class_report_nb = classification_report(y_test, y_pred_nb)

print("Naive Bayes Classifier Evaluation:")
print(f"Accuracy: {accuracy_nb:.4f}")
print("Confusion Matrix:")
print(conf_matrix_nb)
print("Classification Report:")
print(class_report_nb)
print(f"Precision: {precision_nb:.4f}")
print(f"Recall: {recall_nb:.4f}")
print(f"F1 Score: {f1_nb:.4f}")
print()

# Linear Regression Performance
lr_model = LinearRegression()
lr_model.fit(X_train_res, y_train_res)
y_pred_lr = lr_model.predict(X_test)

# Evaluate the Linear Regression Model
mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print()

# Visualize Actual vs Predicted values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), y_pred_lr, color='red', label='Predicted', alpha=0.7)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel(target_column)
plt.legend()
plt.show()

# LGBMRegressor, XGBRegressor, and CatBoostRegressor
lgbm = LGBMRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
catboost = CatBoostRegressor(random_state=42, verbose=0)

lgbm.fit(X_train_res, y_train_res)
xgb.fit(X_train_res, y_train_res)
catboost.fit(X_train_res, y_train_res)

predictions_lgbm = lgbm.predict(X_test)
predictions_xgb = xgb.predict(X_test)
predictions_catboost = catboost.predict(X_test)

mae_lgbm = mean_absolute_error(y_test, predictions_lgbm)
mse_lgbm = mean_squared_error(y_test, predictions_lgbm)
r2_lgbm = r2_score(y_test, predictions_lgbm)

mae_xgb = mean_absolute_error(y_test, predictions_xgb)
mse_xgb = mean_squared_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)

mae_catboost = mean_absolute_error(y_test, predictions_catboost)
mse_catboost = mean_squared_error(y_test, predictions_catboost)
r2_catboost = r2_score(y_test, predictions_catboost)

print('LightGBM:')
print(f'MAE: {mae_lgbm:.4f}')
print(f'MSE: {mse_lgbm:.4f}')
print(f'R-squared: {r2_lgbm:.4f}')
print('------')
print()
print('XGBoost:')
print(f'MAE: {mae_xgb:.4f}')
print(f'MSE: {mse_xgb:.4f}')
print(f'R-squared: {r2_xgb:.4f}')
print('------')
print()
print('CatBoost:')
print(f'MAE: {mae_catboost:.4f}')
print(f'MSE: {mse_catboost:.4f}')
print(f'R-squared: {r2_catboost:.4f}')
print()

# Visualize Actual vs Predicted values for LightGBM
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), predictions_lgbm, color='red', label='Predicted', alpha=0.7)
plt.title("LightGBM: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel(target_column)
plt.legend()
plt.show()

# Visualize Actual vs Predicted values for XGBoost
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), predictions_xgb, color='red', label='Predicted', alpha=0.7)
plt.title("XGBoost: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel(target_column)
plt.legend()
plt.show()

# Visualize Actual vs Predicted values for CatBoost
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_test)), predictions_catboost, color='red', label='Predicted', alpha=0.7)
plt.title("CatBoost: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel(target_column)
plt.legend()
plt.show()

# Visualize the confusion matrix
plt.figure(figsize=(12, 10))
plt.imshow(conf_matrix_nb, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Naive Bayes Confusion Matrix", fontsize=14, pad=20)
plt.colorbar()
tick_marks = np.arange(len(target_encoder.classes_))
class_names = target_encoder.inverse_transform(tick_marks)
plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
plt.yticks(tick_marks, class_names, fontsize=8)
plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
plt.ylabel("True Label", fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

# Random Forest
if 'Statusi' not in csv_data_no_outliers.columns:
    raise ValueError("Data must contain 'Statusi' column")

X = csv_data_no_outliers.select_dtypes(include=['number']).drop(columns=['Statusi']).fillna(0)
y = csv_data_no_outliers['Statusi']
unique_classes = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(14, 7))


class_acc = {cls: accuracy_score(y_test[y_test == cls], y_pred[y_test == cls]) 
             for cls in unique_classes}
bars = plt.bar([str(c) for c in class_acc.keys()], class_acc.values(), color='skyblue')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}', ha='center', va='bottom')

overall_acc = accuracy_score(y_test, y_pred)
plt.axhline(overall_acc, color='red', linestyle='--', 
            label=f'Overall Accuracy ({overall_acc:.1%})')

plt.title('Prediction Accuracy by Statusi Value - Random Forest', fontsize=16, pad=20)
plt.xlabel('Statusi Value', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.1)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix', fontsize=16, pad=20)
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i,j] > 0:
            plt.text(j, i, str(cm[i,j]), 
                    ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

plt.xticks(ticks=range(len(unique_classes)), labels=unique_classes)
plt.yticks(ticks=range(len(unique_classes)), labels=unique_classes)
plt.xlabel('Predicted Statusi', fontsize=14)
plt.ylabel('Actual Statusi', fontsize=14)
plt.tight_layout()
plt.show()

# 8. Print metrics
print(f"\n=== Model Evaluation ===")
print(f"Overall Accuracy: {overall_acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
precision_rf = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nRandom Forest Additional Metrics:")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

# 9. Feature Importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importances.head(10).to_string(index=False))