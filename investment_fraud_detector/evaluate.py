from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.model_evaluation import add_training_stats, evaluate_models, pretty_model_name, save_evaluation_results
from src.model_training import MODEL_CONFIGS, create_vectorizer, train_models
from src.preprocessing import load_dataset, split_dataset
from src.visualization import (
    plot_confusion_matrix,
    plot_fraud_word_frequency,
    plot_label_distribution,
    plot_model_comparison,
    plot_tfidf_top_words,
)


DATA_PATH = Path("data/messages.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def calculate_cross_validation_scores(x, y):
    scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_id, model_config in MODEL_CONFIGS.items():
        pipeline = Pipeline(
            [
                ("tfidf", create_vectorizer()),
                ("model", clone(model_config)),
            ]
        )
        cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        scores[model_id] = cv_scores.mean()
    return scores


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)

    vectorizer, models, training_stats = train_models(
        x_train,
        y_train,
        MODELS_DIR,
        save=False,
        return_stats=True,
    )
    metrics_df, class_metrics_df, reports, predictions = evaluate_models(models, vectorizer, x_test, y_test)
    metrics_df = add_training_stats(metrics_df, training_stats)
    cv_scores = calculate_cross_validation_scores(df["clean_text"], df["label"])
    metrics_df["CV F1-score"] = metrics_df["Model ID"].map(cv_scores)
    metrics_df = metrics_df.sort_values("F1-score", ascending=False)
    best_model_id, cm, labels = save_evaluation_results(
        metrics_df,
        class_metrics_df,
        reports,
        y_test,
        predictions,
        RESULTS_DIR,
    )

    plot_label_distribution(df, RESULTS_DIR)
    plot_model_comparison(metrics_df, RESULTS_DIR)
    plot_confusion_matrix(cm, labels, pretty_model_name(best_model_id), RESULTS_DIR)
    plot_fraud_word_frequency(df, RESULTS_DIR)
    plot_tfidf_top_words(vectorizer, vectorizer.transform(df["clean_text"]), RESULTS_DIR)

    print("Evaluation completed.")
    print(metrics_df.to_string(index=False))
    print(f"Best model by F1-score: {pretty_model_name(best_model_id)}")
    print(f"Results saved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
