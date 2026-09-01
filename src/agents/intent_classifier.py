"""Local intent classifier that replaces the LLM classification call.

Loads the scikit-learn pipeline trained by scripts/train_intent_classifier.py
and classifies a message into one of the orchestrator's intents. It runs
locally with no API key, and returns a confidence score the orchestrator uses
to decide when to fall back to the LLM.
"""

from pathlib import Path

import joblib

CLASSIFIER_PATH = Path(__file__).parent / "artifacts" / "intent_classifier.joblib"


def load_classifier():
    """Load the trained intent-classifier pipeline from disk.

    Returns:
        A fitted scikit-learn pipeline (TF-IDF + MLP) that maps a message to an
        intent label.
    """
    return joblib.load(CLASSIFIER_PATH)


def classify(classifier, message: str) -> tuple[str, float]:
    """Predict the intent of a message and the model's confidence.

    Args:
        classifier: The pipeline returned by load_classifier().
        message: The user's message text.

    Returns:
        A (intent, confidence) tuple, where intent is one of the trained labels
        and confidence is the top class probability, from 0 to 1.
    """
    probabilities = classifier.predict_proba([message])[0]
    best = probabilities.argmax()
    intent = str(classifier.classes_[best])
    confidence = float(probabilities[best])
    return intent, confidence
