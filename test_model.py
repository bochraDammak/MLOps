import pandas as pd

from model_pipeline import baseline_accuracy, get_model


def test_baseline_accuracy():
    y = pd.Series([0, 0, 1, 1])
    assert baseline_accuracy(y) == 0.5


def test_get_model_unknown():
    try:
        get_model("inconnu")
        assert False, "Une ValueError était attendue"
    except ValueError:
        pass
