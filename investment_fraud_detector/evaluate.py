from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_evaluation import (
    add_training_stats,
    evaluate_models,
    pretty_model_name,
    save_error_analysis,
    save_evaluation_results,
)
from src.model_training import MODEL_CONFIGS, create_vectorizer, train_models
from src.preprocessing import load_dataset, split_train_validation_test
from src.visualization import (
    plot_baseline_comparison,
    plot_confusion_matrix,
    plot_data_cleaning_comparison,
    plot_fraud_word_frequency,
    plot_label_distribution,
    plot_model_top_features,
    plot_model_comparison,
    plot_split_distribution,
    plot_text_length_distribution,
    plot_tfidf_top_words,
)


DATA_PATH = Path("data/messages.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RAW_DATA_PATH = Path("data/messages_raw.csv")


def calculate_cross_validation_scores(x, y, groups):
    scores = {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for model_id, model_config in MODEL_CONFIGS.items():
        pipeline = Pipeline(
            [
                ("tfidf", create_vectorizer()),
                ("model", clone(model_config)),
            ]
        )
        cv_scores = cross_val_score(
            pipeline,
            x,
            y,
            groups=groups,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        scores[model_id] = cv_scores.mean()
    return scores


def metadata_features(df):
    text = df["text"].astype(str)
    return pd.DataFrame(
        {
            "characters": text.str.len(),
            "words": text.str.split().str.len(),
            "digits": text.str.count(r"\d"),
            "exclamations": text.str.count("!"),
            "percent_signs": text.str.count("%"),
            "uppercase": text.str.count(r"[A-Z]"),
            "urls": text.str.count(r"http|www"),
        },
        index=df.index,
    )


def evaluate_metadata_baseline(df, y_train, y_test):
    features = metadata_features(df)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(features.loc[y_train.index], y_train)
    predicted = model.predict(features.loc[y_test.index])
    row = {
        "Method": "Metadata-only Logistic Regression",
        "Accuracy": accuracy_score(y_test, predicted),
        "Macro F1": f1_score(y_test, predicted, average="macro"),
        "Fraud Recall": recall_score(y_test, predicted, labels=["Fraud"], average="macro", zero_division=0),
    }
    pd.DataFrame([row]).to_csv(RESULTS_DIR / "metadata_baseline.csv", index=False)
    return row


def save_cleaning_comparison(clean_df):
    frames = []
    for stage, path in [("Before cleaning", RAW_DATA_PATH), ("After cleaning", DATA_PATH)]:
        current = pd.read_csv(path)
        current["length"] = current["text"].astype(str).str.len()
        summary = current.groupby("label").agg(Count=("text", "size"), Mean_Length=("length", "mean")).reset_index()
        summary.insert(0, "Stage", stage)
        frames.append(summary)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "data_cleaning_comparison.csv", index=False)
    return comparison


def save_top_features(vectorizer, models, top_n=12):
    model = models["logistic_regression"]
    features = vectorizer.get_feature_names_out()
    rows = []
    for class_index, label in enumerate(model.classes_):
        indices = model.coef_[class_index].argsort()[-top_n:][::-1]
        rows.extend(
            {"Label": label, "Feature": features[index], "Weight": model.coef_[class_index][index]}
            for index in indices
        )
    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(RESULTS_DIR / "model_top_features.csv", index=False)
    plot_model_top_features(feature_df, RESULTS_DIR)
    return feature_df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    partitions = split_train_validation_test(df)
    train_df = partitions["train"]
    validation_df = partitions["validation"]
    test_df = partitions["test"]

    split_distribution = (
        df.groupby(["split", "label"])
        .size()
        .rename("Count")
        .reset_index()
    )
    split_distribution["Percentage of Full Dataset"] = split_distribution["Count"] / len(df)
    split_distribution.to_csv(RESULTS_DIR / "split_distribution.csv", index=False)

    test_composition = (
        test_df.groupby(["label", "source"])
        .size()
        .rename("Count")
        .reset_index()
    )
    test_composition.to_csv(RESULTS_DIR / "test_set_composition.csv", index=False)

    vectorizer, models, training_stats = train_models(
        train_df["clean_text"],
        train_df["label"],
        MODELS_DIR,
        save=False,
        return_stats=True,
    )
    validation_metrics, _, _, _ = evaluate_models(
        models,
        vectorizer,
        validation_df["clean_text"],
        validation_df["label"],
    )
    validation_metrics.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
    selected_model_id = validation_metrics.iloc[0]["Model ID"]

    metrics_df, class_metrics_df, reports, predictions = evaluate_models(
        models,
        vectorizer,
        test_df["clean_text"],
        test_df["label"],
    )
    metrics_df = add_training_stats(metrics_df, training_stats)
    cv_scores = calculate_cross_validation_scores(
        train_df["clean_text"],
        train_df["label"],
        train_df["template_group"],
    )
    metrics_df["CV F1-score"] = metrics_df["Model ID"].map(cv_scores)
    metrics_df = metrics_df.sort_values("F1-score", ascending=False)
    selection_summary = (
        validation_metrics[["Model", "Model ID", "F1-score", "Fraud Recall"]]
        .rename(
            columns={
                "F1-score": "Validation Macro F1",
                "Fraud Recall": "Validation Fraud Recall",
            }
        )
        .merge(
            metrics_df[["Model ID", "F1-score", "Fraud Recall", "CV F1-score"]],
            on="Model ID",
        )
        .rename(
            columns={
                "F1-score": "Test Macro F1",
                "Fraud Recall": "Test Fraud Recall",
                "CV F1-score": "Training Grouped CV F1",
            }
        )
    )
    selection_summary["Selected on Validation"] = selection_summary["Model ID"] == selected_model_id
    selection_summary.to_csv(RESULTS_DIR / "model_selection_summary.csv", index=False)
    _, _, _ = save_evaluation_results(
        metrics_df,
        class_metrics_df,
        reports,
        test_df["label"],
        predictions,
        RESULTS_DIR,
    )
    labels = [label for label in ["Normal", "Suspicious", "Fraud"] if label in set(test_df["label"])]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_df["label"], predictions[selected_model_id], labels=labels)
    error_df = save_error_analysis(df, test_df["label"], models, vectorizer, predictions, RESULTS_DIR)
    metadata_row = evaluate_metadata_baseline(df, train_df["label"], test_df["label"])
    cleaning_comparison = save_cleaning_comparison(df)
    save_top_features(vectorizer, models)

    baseline_comparison = pd.concat(
        [
            pd.DataFrame([metadata_row])[["Method", "Macro F1"]],
            metrics_df[["Model", "F1-score"]].rename(columns={"Model": "Method", "F1-score": "Macro F1"}),
        ],
        ignore_index=True,
    )
    baseline_comparison.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)

    plot_label_distribution(df, RESULTS_DIR)
    plot_split_distribution(split_distribution, RESULTS_DIR)
    plot_text_length_distribution(df, RESULTS_DIR)
    plot_model_comparison(metrics_df, RESULTS_DIR)
    plot_confusion_matrix(cm, labels, pretty_model_name(selected_model_id), RESULTS_DIR)
    plot_fraud_word_frequency(df, RESULTS_DIR)
    plot_tfidf_top_words(vectorizer, vectorizer.transform(df["clean_text"]), RESULTS_DIR)
    plot_baseline_comparison(baseline_comparison, RESULTS_DIR)
    plot_data_cleaning_comparison(cleaning_comparison, RESULTS_DIR)

    print("Evaluation completed.")
    print("Fixed split distribution:")
    print(split_distribution.to_string(index=False))
    print(f"Selected on validation set: {pretty_model_name(selected_model_id)}")
    print("Model selection summary:")
    print(selection_summary.to_string(index=False))
    print("Test set composition:")
    print(test_composition.to_string(index=False))
    print(f"Saved {len(error_df)} misclassified model-message pairs.")
    print("Metadata baseline:")
    print(pd.DataFrame([metadata_row]).to_string(index=False))
    print("Cleaning comparison:")
    print(cleaning_comparison.to_string(index=False))
    print(metrics_df.to_string(index=False))
    print(f"Final test results reported for validation-selected model: {pretty_model_name(selected_model_id)}")
    print(f"Results saved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
