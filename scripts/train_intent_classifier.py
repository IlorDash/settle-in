"""Train and save SettleIn's intent classifier from the generated data.

A lightweight scikit-learn pipeline (TF-IDF features -> a small multi-layer
perceptron) so the bot loads it instantly, with no heavy deep-learning runtime.
The whole pipeline is saved as one file the bot loads at startup. Uses only the
standard library plus scikit-learn (no pandas), keeping dependencies minimal.
"""

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/intent_training.csv")
ARTIFACT_PATH = Path("src/agents/artifacts/intent_classifier.joblib")


def load_data(path, keep_intents=None):
    """Read a labeled CSV into parallel lists, optionally keeping only some intents."""
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if keep_intents and row["intent"] not in keep_intents:
                continue
            texts.append(row["text"])
            labels.append(row["intent"])
    return texts, labels


def main() -> None:
    """Split the data, train the pipeline, report scores, and save it."""
    texts, labels = load_data(DATA_PATH)

    # Train / validation / test split, stratified to preserve class balance.
    # First peel off 30% as a temp pool, then halve it into validation and test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # One pipeline: TF-IDF -> small MLP (one hidden layer of 64 units).
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=2)),
            (
                "mlp",
                MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)

    print(f"validation accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    print(f"test accuracy:       {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print("Saved pipeline to", ARTIFACT_PATH.resolve())


if __name__ == "__main__":
    main()
