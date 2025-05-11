from data_loader import load_data
from preprocessing import encode_categorical, remove_outliers, apply_smote
from visualization import plot_distributions, plot_confusion_matrix_percent, \
    plot_per_class_accuracy, plot_db_index, plot_regression_scatter, plot_training_history
from models import *
from evaluation import *
from sklearn.model_selection import train_test_split
import numpy as np

numerical_cols = [
    "Viti", "Muaji", "Tatimpaguesit",
    "Blerjet dhe importet investive pa TVSH",
    "Blerjet dhe importet investive me TVSH jo te zbritshme",
    "Importet investive me norme 18%",
    "Importet investive me norme 8%",
    "Blerjet investive vendore me norme 18%",
    "Blerjet investive vendore me norme 8%"
]

categorical_cols = ['Pershkrimi', 'Statusi', 'Komuna']
target_column = 'Statusi'

data = load_data("atk-investimet-tvsh.csv")

data, encoders = encode_categorical(data, categorical_cols)

plot_distributions(data, numerical_cols, "(Original)", show_plot=False)

data_clean = remove_outliers(data, numerical_cols)
data_clean.to_csv("Investimet-2024-cleaned-no-outliers.csv", index=False)

plot_distributions(data_clean, numerical_cols, "(Cleaned)", show_plot=False)


X, y = data_clean.drop(columns=target_column), data_clean[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_res, y_train_res = apply_smote(X_train, y_train)

models_dict = {
    "Naive Bayes": train_naive_bayes(X_train_res, y_train_res),
    "Linear Regression": train_linear_regression(X_train_res, y_train_res),
    "LightGBM": train_lgbm(X_train_res, y_train_res),
    "XGBoost": train_xgboost(X_train_res, y_train_res),
    "CatBoost": train_catboost(X_train_res, y_train_res),
    "Random Forest": train_random_forest(X_train_res, y_train_res),
    "Neural Network": train_mlp(X_train_res, y_train_res),
    "K‑Means": train_kmeans(X_train_res, n_clusters=len(encoders[target_column].classes_))
}

for name, model in models_dict.items():
    print(f"\n {name} Evaluation")

    if name == "K‑Means":
        cluster_labels = model.predict(X_test)
        evaluate_clustering(X_test, cluster_labels, y_true=y_test)
        plot_db_index(X_test, k_min=2, k_max=10)
        continue

    if name == "Linear Regression":
        preds = model.predict(X_test)
        evaluate_regression(y_test, preds)
        plot_regression_scatter(y_test, preds)


    elif name in ["LightGBM", "XGBoost", "CatBoost"]:
        preds = model.predict(X_test)
        preds_class = np.clip(np.round(preds).astype(int), y_train_res.min(), y_train_res.max())

        evaluate_classification(y_test, preds_class)
        plot_per_class_accuracy(y_test, preds_class, encoders[target_column].classes_, name)
        plot_confusion_matrix_percent(y_test, preds_class, encoders[target_column].classes_, name + " Confusion Matrix")

        # Neural Network
    if name == "Neural Network":
            # predict probabilities → class indices
        probs = model.predict(X_test)
        preds = np.argmax(probs, axis=1)

            # detailed report
        print(classification_report(
            y_test,
            preds,
            target_names=encoders[target_column].classes_
            ))

            # summary metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")

            # and your usual plots
        plot_per_class_accuracy(
            y_test, preds,
            encoders[target_column].classes_,
            title="MLP Accuracy by Class"
            )
        plot_confusion_matrix_percent(
            y_test, preds,
            encoders[target_column].classes_,
            title="MLP Confusion Matrix"
            )
        plot_training_history("Neural Network")
        continue

    else:
        preds = model.predict(X_test)
        evaluate_classification(y_test, preds)

        if name == "Naive Bayes":
            plot_per_class_accuracy(y_test, preds, encoders[target_column].classes_, name)
            plot_confusion_matrix_percent(y_test, preds, encoders[target_column].classes_,
                                          "Naive Bayes Confusion Matrix")

        elif name == "Random Forest":
            plot_per_class_accuracy(y_test, preds, encoders[target_column].classes_, name)
            plot_confusion_matrix_percent(y_test, preds, encoders[target_column].classes_,
                                          "Random Forest Confusion Matrix")


