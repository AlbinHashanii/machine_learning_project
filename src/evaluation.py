from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_regression(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")

def evaluate_classification(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

