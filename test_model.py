import subprocess
import sys

import pandas as pd

from model_pipeline import baseline_accuracy, get_model

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_tool(args):
    return subprocess.run([sys.executable, "-m"] + args, capture_output=True, text=True)


# ---------- Tests sur le code ML ----------
def test_baseline_accuracy():
    y = pd.Series([0, 0, 1, 1])
    assert baseline_accuracy(y) == 0.5


def test_get_model_unknown():
    try:
        get_model("inconnu")
        assert False, "Une ValueError était attendue"
    except ValueError:
        pass


# ---------- Tests formatage, qualité, sécurité ----------
def test_formatting_black():
    result = run_tool(["black", "--check", "--line-length", "100"] + FILES)
    assert result.returncode == 0, result.stderr


def test_quality_flake8():
    result = run_tool(["flake8", "--max-line-length=100"] + FILES)
    assert result.returncode == 0, result.stdout


def test_security_bandit():
    result = run_tool(["bandit", "-ll", "-q"] + FILES)
    assert result.returncode == 0, result.stdout
