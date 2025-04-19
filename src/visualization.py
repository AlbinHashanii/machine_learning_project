import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, silhouette_samples, davies_bouldin_score


def plot_distributions(df: pd.DataFrame, cols: list, title_suffix: str = "", show_plot: bool = True):
    for col in cols:
        if not show_plot:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(df[col].median(), color='orange', linestyle='-', label='Median')
        plt.title(f"{col} Distribution {title_suffix} (Skewness: {skew(df[col], nan_policy='omit'):.2f})")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_labels, title="Prediction Accuracy by Class"):
    classes = np.unique(y_true)
    acc_per_class = {
        label: accuracy_score(y_true[y_true == label], y_pred[y_true == label])
        for label in classes
    }

    values = list(acc_per_class.values())
    labels = [class_labels[c] if isinstance(class_labels[0], str) else str(c) for c in classes]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, values, color='skyblue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=8)

    overall_acc = accuracy_score(y_true, y_pred)
    plt.axhline(overall_acc, color='red', linestyle='--', label=f'Overall Accuracy ({overall_acc:.1%})')

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Statusi Value", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


def plot_confusion_matrix_percent(y_true, y_pred, class_names, title="Confusion Matrix with Percentages"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value > 0:
                plt.text(j, i, f'{value}', ha='center', va='center',
                         color='white' if value > cm.max() * 0.5 else 'black', fontsize=7)

    plt.xlabel("Predicted Statusi", fontsize=14)
    plt.ylabel("Actual Statusi", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def plot_db_index(X, k_min=2, k_max=10):
    ks = list(range(k_min, k_max+1))
    db_scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        db_scores.append(davies_bouldin_score(X, km.labels_))
    plt.figure(figsize=(8, 4))
    plt.plot(ks, db_scores, marker='o')
    plt.title("Davies‑Bouldin Index vs. Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies‑Bouldin Index") # lower is better
    plt.xticks(ks)
    plt.tight_layout()
    plt.show()

def plot_regression_scatter(y_true, y_pred, title="Regression: Actual vs Predicted"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.tight_layout()
    plt.show()