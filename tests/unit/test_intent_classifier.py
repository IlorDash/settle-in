from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.intent_classifier import classify

INTENTS = ["knowledge_question", "out_of_scope", "translation"]


def _fake_classifier(probabilities):
    """A stand-in pipeline whose predict_proba returns the given probabilities."""
    classifier = MagicMock()
    classifier.classes_ = np.array(INTENTS)
    classifier.predict_proba.return_value = np.array([probabilities])
    return classifier


def test_classify_returns_the_highest_probability_intent():
    classifier = _fake_classifier([0.1, 0.2, 0.7])
    intent, _ = classify(classifier, "translate hello to serbian")
    assert intent == "translation"


def test_classify_returns_the_top_probability_as_confidence():
    classifier = _fake_classifier([0.1, 0.2, 0.7])
    _, confidence = classify(classifier, "translate hello to serbian")
    assert confidence == pytest.approx(0.7)


def test_classify_passes_the_message_to_the_model():
    classifier = _fake_classifier([1.0, 0.0, 0.0])
    classify(classifier, "how do I get a residence permit")
    classifier.predict_proba.assert_called_once_with(
        ["how do I get a residence permit"]
    )


def test_classify_returns_a_python_str_and_float():
    classifier = _fake_classifier([0.0, 0.9, 0.1])
    intent, confidence = classify(classifier, "tell me a joke")
    assert isinstance(intent, str) and isinstance(confidence, float)
