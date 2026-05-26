from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix

from src.preprocessing import LABEL_ORDER, clean_text


sns.set_theme(style="whitegrid")


def _save_current_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_label_distribution(df, results_dir="results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    counts = df["label"].value_counts().reindex(LABEL_ORDER).dropna()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", legend=False)
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    _save_current_plot(results_dir / "label_distribution.png")


def plot_model_comparison(metrics_df, results_dir="results"):
    results_dir = Path(results_dir)
    plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set1")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    _save_current_plot(results_dir / "model_comparison.png")


def plot_confusion_matrix(cm, labels, best_model_name, results_dir="results"):
    results_dir = Path(results_dir)
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    _save_current_plot(results_dir / "confusion_matrix_best_model.png")


def plot_confusion_matrices_all_models(y_test, predictions, labels, model_names, results_dir="results"):
    results_dir = Path(results_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for index, (model_id, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        ax = axes[index]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(model_names.get(model_id, model_id))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for empty_index in range(len(predictions), len(axes)):
        axes[empty_index].axis("off")

    fig.suptitle("Confusion Matrix Comparison - All Models", fontsize=14)
    _save_current_plot(results_dir / "confusion_matrix_all_models.png")


def plot_per_class_f1_comparison(class_metrics_df, results_dir="results"):
    results_dir = Path(results_dir)
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=class_metrics_df,
        x="Label",
        y="F1-score",
        hue="Model",
        palette="Set2",
    )
    ax.set_title("Per-Class F1-score Comparison")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_current_plot(results_dir / "per_class_f1_comparison.png")


def plot_fraud_precision_recall_comparison(class_metrics_df, results_dir="results"):
    results_dir = Path(results_dir)
    fraud_df = class_metrics_df[class_metrics_df["Label"] == "Fraud"].copy()
    plot_df = fraud_df.melt(
        id_vars=["Model"],
        value_vars=["Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set1")
    ax.set_title("Fraud Detection Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    _save_current_plot(results_dir / "fraud_precision_recall_comparison.png")


def plot_fraud_word_frequency(df, results_dir="results", top_n=20):
    results_dir = Path(results_dir)
    fraud_text = " ".join(df.loc[df["label"] == "Fraud", "clean_text"].astype(str))
    words = [
        word
        for word in fraud_text.split()
        if word not in ENGLISH_STOP_WORDS and len(word) > 2 and not word.isdigit()
    ]
    counts = Counter(words).most_common(top_n)
    plot_word_bars(counts, "Top Fraud Words", "Frequency", results_dir / "fraud_word_frequency.png")


def plot_tfidf_top_words(vectorizer, x_tfidf, results_dir="results", top_n=20):
    results_dir = Path(results_dir)
    scores = x_tfidf.mean(axis=0).A1
    features = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[-top_n:][::-1]
    data = [(features[index], scores[index]) for index in top_indices]
    plot_word_bars(data, "Top TF-IDF Words", "Mean TF-IDF Score", results_dir / "tfidf_top_words.png")


def plot_word_bars(word_scores, title, ylabel, output_path):
    output_path = Path(output_path)
    words = [item[0] for item in word_scores]
    scores = [item[1] for item in word_scores]

    if not words:
        words = ["No data"]
        scores = [0]

    plot_df = pd.DataFrame({"Word": words, "Score": scores})
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Score", y="Word", hue="Word", palette="viridis", legend=False)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Word")
    _save_current_plot(output_path)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "text": ["Guaranteed profit now", "hello friend"],
            "label": ["Fraud", "Normal"],
        }
    )
    sample["clean_text"] = sample["text"].apply(clean_text)
    plot_label_distribution(sample)
    plot_fraud_word_frequency(sample)
    print("Sample charts saved to results/.")
