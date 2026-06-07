# Investment Scam Message Classification

This project classifies messages into three labels:

- `Normal`
- `Suspicious`
- `Fraud`

It uses TF-IDF text features and traditional machine learning models. No BERT, deep learning, database, or login system is used.

## Project Structure

```text
investment_fraud_detector/
|-- app.py
|-- train.py
|-- evaluate.py
|-- predict.py
|-- data/
|   `-- messages.csv
|-- src/
|   |-- preprocessing.py
|   |-- model_training.py
|   |-- model_evaluation.py
|   |-- predictor.py
|   `-- visualization.py
|-- models/
|-- results/
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Train Final Models With Full Dataset

```bash
python train.py
```

`train.py` trains all four models with the full dataset in `data/messages.csv`.
These model files are used by `predict.py` and `app.py`.

Saved files:

- `models/vectorizer.pkl`
- `models/naive_bayes.pkl`
- `models/logistic_regression.pkl`
- `models/svm.pkl`
- `models/random_forest.pkl`

## Evaluate and Compare Models

```bash
python evaluate.py
```

`evaluate.py` uses grouped train-test splitting for fair model comparison.
Do not evaluate with the full training dataset, because that would test the models on messages they already saw.

The dataset contains fixed group-isolated partitions:

- Training: approximately 70%
- Validation: approximately 20%
- Test: approximately 10%

Models are trained on the training partition, selected using validation Macro
F1, and reported once on the final test partition.

Saved model comparison outputs:

- `results/metrics.csv`
- `results/classification_report.txt`
- `results/test_set_composition.csv`
- `results/split_distribution.csv`
- `results/validation_metrics.csv`
- `results/model_selection_summary.csv`
- `results/metadata_baseline.csv`
- `results/baseline_comparison.csv`
- `results/error_analysis.csv`
- `results/model_top_features.csv`
- `results/data_cleaning_comparison.csv`
- `results/model_comparison.png`
- `results/confusion_matrix_best_model.png`
- `results/baseline_comparison.png`
- `results/model_top_features.png`
- `results/data_cleaning_comparison.png`
- `results/split_distribution.png`

Saved data analysis outputs:

- `results/label_distribution.png`
- `results/text_length_distribution.png`
- `results/fraud_word_frequency.png`
- `results/tfidf_top_words.png`

## Model Strengths and Applications

| Model | Strength | Weakness | Best Use |
| --- | --- | --- | --- |
| Naive Bayes | Fast and simple baseline model | May miss more complex fraud wording | Quick first-pass filtering or classroom baseline comparison |
| Logistic Regression | Stable linear text classifier and easy to explain | Lower grouped holdout F1 in the current experiment | Interpretable linear comparison model |
| SVM | Highest grouped holdout and cross-validation F1 in the current experiment | Its displayed class scores are not calibrated probabilities | Default lightweight screening model |
| Random Forest | Can capture non-linear patterns | Larger, slower, and lower F1 in the current experiment | Non-linear comparison model |

## Single Prediction

```bash
python predict.py "Guaranteed profit with zero risk. Join our VIP crypto group now!"
```

Example output:

```text
Prediction: Fraud
Model Confidence: 92.35%
Needs Review: No
Class Scores:
  Fraud: 92.35%
  Suspicious: 5.10%
  Normal: 2.55%
```

## Streamlit Demo

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The demo includes:

- Message input
- Analyze button
- Predicted label
- Model confidence and class scores
- Low-confidence review status
- Single-message comparison across all four models
- Batch CSV screening and result download
- Model comparison charts
- Model comparison table with Fraud metrics, cross-validation F1, timing, and model size
- Data analysis charts

## Rebuild the Dataset

```bash
python build_dataset.py
```

The rebuild process:

- Keeps original everyday messages as negative examples.
- Removes original fraud messages unrelated to investment fraud.
- Removes suspicious messages containing obvious label-leakage wording.
- Adds longer synthetic investment messages to all three classes.
- Marks every row with `source` and `template_group`.

Synthetic examples are included for controlled model comparison and must be
reported as synthetic data. They should not be presented as real scam reports.

## Evaluation Design

`evaluate.py` uses grouped splitting and grouped five-fold cross-validation.
Messages from the same synthetic scenario family cannot appear in both the
training and test partitions. This produces a more realistic estimate than a
random split of similar synthetic messages.

## Application Layer

The Streamlit application uses only model outputs. It does not use manually
assigned keyword bonuses or a heuristic risk score.

Application features:

- Single-message classification
- Model-derived class scores
- Comparison across all four models
- Low-confidence review flag
- Batch CSV screening and downloadable results
