from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, classification_report, confusion_matrix, silhouette_score, adjusted_rand_score, \
    davies_bouldin_score
import numpy as np
from scipy.stats import mode

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

def evaluate_clustering(X, cluster_labels, y_true=None):
    sil = silhouette_score(X, cluster_labels)
    print(f"Silhouette Score: {sil:.4f}")
    db = davies_bouldin_score(X, cluster_labels)
    print(f"Davies‑Bouldin Index: {db:.4f}")
    if y_true is not None:
        ari = adjusted_rand_score(y_true, cluster_labels)
        print(f"Adjusted Rand Index: {ari:.4f}")


