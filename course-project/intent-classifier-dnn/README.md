# DNN Intent Classifier on CLINC150

Course project for **Deep Neural Networks**. Standalone experiment that lives
in the SettleIn repository but is **not** part of the bot's runtime code.

## Goal

Build a proof-of-concept intent classifier with deep neural networks on the
**CLINC150** benchmark, and evaluate whether it could replace the current
LLM-based intent classifier in the SettleIn thesis project.

Today SettleIn classifies user intent with a GPT-4o-mini API call
(`src/agents/orchestrator.py`). This project asks: can a small, **fast,
deterministic, cost-free** DNN do the job instead — including detecting
**out-of-scope** questions (queries outside supported topics)?

> Honest scope: CLINC150's 150 intents (banking, travel, work…) are NOT
> SettleIn's intents. This validates the *method*; transferring it means
> retraining the same architecture on SettleIn's own labelled intents later.

## Dataset

CLINC150 — 150 intent classes across 10 domains, ~15,000 training samples,
plus dedicated **out-of-scope** examples.
Kaggle: https://www.kaggle.com/datasets/hongtrung/clinc150-dataset

Download the dataset and place the JSON file(s) in `data/` (git-ignored).

## Experiments

| Stage | Model | Purpose |
|-------|-------|---------|
| Baseline | TF-IDF + Dense MLP | Score to beat, uses bag-of-words features |
| Deep 1 | Embedding + GlobalAveragePooling | Learned word embeddings |
| Deep 2 | Embedding + Conv1D / BiLSTM | Sequence-aware deep model |
| Tuning | vary embed dim / dropout / units / LR | Compare validation curves |
| OOS | confidence threshold **and** OOS-as-151st-class | Detect off-topic queries |

Evaluation: accuracy, **macro-F1** (150 classes), per-domain confusion
matrix (150→10 aggregated for readability), and OOS recall vs in-scope
accuracy trade-off.

## Setup (dedicated environment)

Use a separate virtual environment so heavy TensorFlow/numpy pins never
conflict with the bot's dependencies.

```powershell
# from this folder: course-project/intent-classifier-dnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name settlein-dnn --display-name "Python (settlein-dnn)"
jupyter notebook intent_classification_clinc150.ipynb
```

Select the **Python (settlein-dnn)** kernel inside the notebook.

## Structure

```
intent-classifier-dnn/
├── intent_classification_clinc150.ipynb   # the deliverable
├── requirements.txt
├── data/      # CLINC150 JSON (download from Kaggle; git-ignored)
└── models/    # saved .keras artifacts (git-ignored)
```
