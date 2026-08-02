"""Integration tests: load the real trained classifier and check it works.

Skipped automatically if the model artifact has not been trained yet.
"""

import pytest

from src.agents.intent_classifier import (
    CLASSIFIER_PATH,
    classify,
    load_classifier,
)

pytestmark = pytest.mark.skipif(
    not CLASSIFIER_PATH.exists(),
    reason=(
        "trained classifier artifact not present; "
        "run scripts/train_intent_classifier.py"
    ),
)


@pytest.fixture(scope="module")
def classifier():
    return load_classifier()


@pytest.mark.parametrize(
    "message, expected_intent",
    [
        ("Как будет по-сербски спасибо?", "translation"),
        ("How do I get a residence permit in Serbia?", "knowledge_question"),
        ("tell me a joke about cats", "out_of_scope"),
    ],
)
def test_real_classifier_predicts_expected_intent(classifier, message, expected_intent):
    intent, _ = classify(classifier, message)
    assert intent == expected_intent


def test_real_classifier_confidence_is_a_probability(classifier):
    _, confidence = classify(classifier, "Переведи на русский добар дан")
    assert 0.0 <= confidence <= 1.0
