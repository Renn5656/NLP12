from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing import LABEL_ORDER


def evaluate_models(models, vectorizer, x_test, y_test):
    x_test_tfidf = vectorizer.transform(x_test)
    rows = []
    class_metric_rows = []
    reports = {}
    predictions = {}

    labels = [label for label in LABEL_ORDER if label in set(y_test)]

    for model_name, model in models.items():
        start_time = time.perf_counter()
        y_pred = model.predict(x_test_tfidf)
        prediction_time = time.perf_counter() - start_time
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        fraud_report = report_dict.get("Fraud", {"precision": 0, "recall": 0, "f1-score": 0})
        rows.append(
            {
                "Model": pretty_model_name(model_name),
                "Model ID": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Fraud Precision": fraud_report["precision"],
                "Fraud Recall": fraud_report["recall"],
                "Fraud F1-score": fraud_report["f1-score"],
                "Prediction Time (ms/msg)": (prediction_time / len(y_test)) * 1000,
            }
        )
        reports[model_name] = classification_report(
            y_test,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        for label in labels:
            class_metric_rows.append(
                {
                    "Model": pretty_model_name(model_name),
                    "Model ID": model_name,
                    "Label": label,
                    "Precision": report_dict[label]["precision"],
                    "Recall": report_dict[label]["recall"],
                    "F1-score": report_dict[label]["f1-score"],
                    "Support": report_dict[label]["support"],
                }
            )
        predictions[model_name] = y_pred

    metrics_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    class_metrics_df = pd.DataFrame(class_metric_rows)
    return metrics_df, class_metrics_df, reports, predictions


def add_training_stats(metrics_df, training_stats):
    metrics_df = metrics_df.copy()
    metrics_df["Training Time (sec)"] = metrics_df["Model ID"].map(
        lambda model_id: training_stats[model_id]["Training Time (sec)"]
    )
    metrics_df["Model Size (KB)"] = metrics_df["Model ID"].map(
        lambda model_id: training_stats[model_id]["Model Size (KB)"]
    )
    return metrics_df


def save_evaluation_results(metrics_df, class_metrics_df, reports, y_test, predictions, results_dir="results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        for _, row in metrics_df.iterrows():
            model_id = row["Model ID"]
            file.write(f"=== {row['Model']} ===\n")
            file.write(reports[model_id])
            file.write("\n\n")

    best_model_id = metrics_df.iloc[0]["Model ID"]
    labels = [label for label in LABEL_ORDER if label in set(y_test)]
    cm = confusion_matrix(y_test, predictions[best_model_id], labels=labels)
    return best_model_id, cm, labels


def pretty_model_name(model_name):
    return {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "random_forest": "Random Forest",
    }.get(model_name, model_name.replace("_", " ").title())


def model_recommendation(model_name):
    recommendations = {
        "Naive Bayes": "Fast baseline; suitable for quick first-pass filtering.",
        "Logistic Regression": "Recommended main model; stable and explainable.",
        "SVM": "Strong text classifier; good when accuracy matters more than probability output.",
        "Random Forest": "Useful comparison model; larger and less ideal for sparse TF-IDF text.",
    }
    return recommendations.get(model_name, "General comparison model.")
