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
|   |-- risk_scoring.py
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

`evaluate.py` uses an 80/20 train-test split for fair model comparison.
Do not evaluate with the full training dataset, because that would test the models on messages they already saw.

Saved model comparison outputs:

- `results/metrics.csv`
- `results/classification_report.txt`
- `results/model_comparison.png`
- `results/confusion_matrix_best_model.png`

Saved data analysis outputs:

- `results/label_distribution.png`
- `results/fraud_word_frequency.png`
- `results/tfidf_top_words.png`

## Model Strengths and Applications

| Model | Strength | Weakness | Best Use |
| --- | --- | --- | --- |
| Naive Bayes | Fast and simple baseline model | May miss more complex fraud wording | Quick first-pass filtering or classroom baseline comparison |
| Logistic Regression | Stable overall performance and easy to explain | Limited ability to model complex language patterns | Recommended main model for this project |
| SVM | Strong text classification performance | Less direct probability interpretation | When high classification accuracy matters more than probability output |
| Random Forest | Can capture non-linear feature patterns | Large model size and not always ideal for sparse TF-IDF text | Comparison model for explaining trade-offs |

## Single Prediction

```bash
python predict.py "Guaranteed profit with zero risk. Join our VIP crypto group now!"
```

Example output:

```text
Prediction: Fraud
Risk Level: High Risk
Risk Score: 100
Detected Suspicious Keywords: guaranteed profit, zero risk, vip
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
- Risk score
- Risk level
- Detected suspicious keywords
- Human-readable risk explanation
- Risk score breakdown
- Single-message comparison across all four models
- Model comparison charts
- Model comparison table with Fraud metrics, cross-validation F1, timing, and model size
- Data analysis charts

## Risk Scoring

Base scores:

- `Normal`: 15
- `Suspicious`: 40
- `Fraud`: 65

Extra investment scam signal scores:

- Guaranteed profit claim: +8
- No-risk claim: +8
- Unrealistic return: +7
- Crypto or digital asset payment: +5
- VIP or private group lure: +4
- Urgency or scarcity: +4
- Advance fee or withdrawal obstacle: +10
- Fake platform claim: +5
- Relationship investment pattern: +5
- High-risk payment method: +7

Maximum score is 100.

Combination bonuses:

- Advance fee or withdrawal obstacle + crypto/payment signal: +8
- Guaranteed profit claim + no-risk claim: +5
- Relationship investment pattern + crypto signal: +5

Risk level:

- `0-25`: Very Low Risk
- `26-35`: Low Risk
- `36-60`: Medium Risk
- `61-80`: High Risk
- `81-100`: Very High Risk
