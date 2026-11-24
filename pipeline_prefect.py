import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

# Seuls ces fichiers sont analysés (qualité et sécurité)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_PATH = "adult.data.csv"
MODEL_PATH = "model.joblib"


def run(cmd):
    """Lance une commande et lève une erreur si elle échoue."""
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------- Tasks : environnement et code ----------
@task(name="Installer les dépendances")
def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run([sys.executable, "-m", "black", "--line-length", "100"] + FILES)


@task(name="Qualité du code")
def quality_code():
    run([sys.executable, "-m", "flake8", "--max-line-length=100"] + FILES)


@task(name="Sécurité du code")
def security_code():
    run([sys.executable, "-m", "bandit", "-ll", "-q"] + FILES)


@task(name="Tests unitaires")
def unit_tests():
    run([sys.executable, "-m", "pytest", "-q", "test_model.py"])


# ---------- Tasks : données et modèle ----------
@task(name="Préparation des données")
def prepare_task():
    return prepare_data(DATA_PATH)


@task(name="Entraînement du modèle")
def train_task(X_train, y_train):
    return train_model(X_train, y_train, model_type="xgboost")


@task(name="Sauvegarde du modèle")
def save_task(model):
    return save_model(model, MODEL_PATH)


@task(name="Chargement du modèle")
def load_task():
    return load_model(MODEL_PATH)


@task(name="Évaluation du modèle")
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ---------- Flows ----------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    code_flow()
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    evaluate_task(model, X_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
