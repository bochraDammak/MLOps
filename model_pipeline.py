"""
model_pipeline.py
=================
Pipeline ML modulaire pour le projet « Adult Census Income »
(classification binaire : income > 50K ou <= 50K).

Ce module est issu de la modularisation du notebook Data_Adult_mlops.ipynb.
Chaque étape du notebook est encapsulée dans une fonction réutilisable :

    prepare_data()   -> chargement + nettoyage + transformation + split
    train_model()    -> entraînement d'un modèle (5 algorithmes disponibles)
    evaluate_model() -> métriques de performance sur le jeu de test
    save_model()     -> sauvegarde avec joblib
    load_model()     -> chargement avec joblib

Fonctions utilitaires : baseline_accuracy(), compare_models().
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# Constantes (issues du notebook)
# --------------------------------------------------------------------------
RANDOM_STATE = 42

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# Colonnes numériques à normaliser (Partie 2 du notebook)
NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# Colonnes avec valeurs manquantes codées "?" (Partie 1 du notebook)
MISSING_COLS = ["workclass", "occupation", "native_country"]

# Variables retenues en Partie 3 (cohérentes avec l'ODS)
SELECTED_FEATURES = ["age", "education_num", "hours_per_week"]

TARGET = "income"

MODEL_CHOICES = ["logreg", "tree", "forest", "xgboost", "knn"]


# --------------------------------------------------------------------------
# 1. PRÉPARATION DES DONNÉES
# --------------------------------------------------------------------------
def load_data(path):
    """Charge le fichier CSV Adult (sans en-tête) dans un DataFrame.

    Args:
        path (str): chemin du fichier CSV (ex. "adult.data.csv").

    Returns:
        pd.DataFrame: données brutes avec les 15 colonnes nommées.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le nombre de colonnes est incorrect.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True)

    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{len(COLUMNS)} colonnes attendues, {df.shape[1]} trouvées.")
    return df


def clean_data(df):
    """Nettoie les données (Partie 1 du notebook).

    - Les doublons sont conservés (décision du notebook : impact négligeable).
    - Les valeurs manquantes codées "?" sont remplacées par la catégorie
      "Unknown" pour workclass, occupation et native_country.

    Args:
        df (pd.DataFrame): données brutes.

    Returns:
        pd.DataFrame: données nettoyées (sans aucune valeur manquante).
    """
    df = df.replace("?", np.nan)
    for col in MISSING_COLS:
        df[col] = df[col].fillna("Unknown")
    return df


def transform_data(df):
    """Transforme les données (Partie 2 du notebook).

    - Encode les variables catégorielles avec LabelEncoder.
    - Encode la cible : 1 si ">50K", 0 sinon.
    - Normalise les variables numériques avec StandardScaler.

    Args:
        df (pd.DataFrame): données nettoyées.

    Returns:
        tuple: (df transformé, scaler ajusté)
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != TARGET:
            df[col] = LabelEncoder().fit_transform(df[col])

    df[TARGET] = df[TARGET].apply(lambda x: 1 if ">50K" in str(x) else 0)

    scaler = StandardScaler()
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    return df, scaler


def prepare_data(path, features=None, test_size=0.2, random_state=RANDOM_STATE):
    """Charge et prétraite les données, puis sépare train / test.

    Enchaîne : load_data -> clean_data -> transform_data -> sélection des
    variables -> train_test_split.

    Args:
        path (str): chemin du fichier CSV.
        features (list[str] | None): variables à utiliser.
            Par défaut : SELECTED_FEATURES (age, education_num, hours_per_week).
        test_size (float): part du jeu de test (0.2 = 20 %).
        random_state (int): graine pour un découpage reproductible.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    features = features or SELECTED_FEATURES

    df = load_data(path)
    df = clean_data(df)
    df, _ = transform_data(df)

    unknown = [f for f in features if f not in df.columns]
    if unknown:
        raise ValueError(f"Variables inconnues : {unknown}")

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[prepare_data] Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 2. ENTRAÎNEMENT
# --------------------------------------------------------------------------
def _import_xgboost():
    """Importe XGBClassifier uniquement quand il est nécessaire (import paresseux)."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost n'est pas installé. Installez-le (pip install xgboost-cpu) "
            "ou utilisez un autre modèle : --model logreg"
        ) from exc
    return XGBClassifier


def get_model(model_type="xgboost"):
    """Retourne un modèle non entraîné selon son nom.

    Args:
        model_type (str): "logreg", "tree", "forest", "xgboost" ou "knn".

    Returns:
        Un estimateur scikit-learn / XGBoost avec les hyperparamètres du notebook.
    """
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if model_type == "tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE)
    if model_type == "forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_type == "xgboost":
        XGBClassifier = _import_xgboost()
        return XGBClassifier(
            learning_rate=0.1,
            max_depth=3,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
    raise ValueError(f"Modèle inconnu '{model_type}'. Choix : {MODEL_CHOICES}")


def train_model(X_train, y_train, model_type="xgboost", tune=False):
    """Entraîne un modèle.

    Args:
        X_train (pd.DataFrame): variables d'entraînement.
        y_train (pd.Series): cible d'entraînement.
        model_type (str): "logreg", "tree", "forest", "xgboost" ou "knn".
        tune (bool): si True et model_type == "xgboost", optimise les
            hyperparamètres avec RandomizedSearchCV (score = AUC, cv=3,
            uniquement sur le train).

    Returns:
        Le modèle entraîné.
    """
    if tune:
        if model_type != "xgboost":
            raise ValueError("L'optimisation (tune=True) n'est prévue que pour 'xgboost'.")

        XGBClassifier = _import_xgboost()
        param_dist = {
            "learning_rate": [0.3, 0.1, 0.05, 0.01],
            "max_depth": [2, 3, 4, 5, 7],
            "n_estimators": [100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.5, 1],
            "reg_alpha": [0, 1, 10],
            "reg_lambda": [1, 5, 10],
        }
        search = RandomizedSearchCV(
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE
            ),
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        print("[train_model] Meilleurs hyperparamètres :", search.best_params_)
        print(f"[train_model] Meilleur AUC (CV) : {search.best_score_:.4f}")
        return search.best_estimator_

    model = get_model(model_type)
    model.fit(X_train, y_train)
    print(f"[train_model] Modèle '{model_type}' entraîné.")
    return model


# --------------------------------------------------------------------------
# 3. ÉVALUATION
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Évalue un modèle sur le jeu de test.

    Args:
        model: modèle entraîné (doit avoir predict_proba).
        X_test (pd.DataFrame): variables de test.
        y_test (pd.Series): cible de test.
        threshold (float): seuil de décision pour classer ">50K".

    Returns:
        dict: Accuracy, Précision, Rappel, F1, AUC.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Précision": precision_score(y_test, y_pred, zero_division=0),
        "Rappel": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
    }
    for name, value in metrics.items():
        print(f"[evaluate_model] {name:<10}: {value:.3f}")
    return metrics


def baseline_accuracy(y_test):
    """Accuracy d'un « modèle » qui prédit toujours la classe majoritaire (<=50K).

    Args:
        y_test (pd.Series): cible de test (0/1).

    Returns:
        float: accuracy de la baseline.
    """
    return 1 - y_test.mean()


def compare_models(X_train, y_train, model_types=None, n_splits=5):
    """Compare plusieurs modèles par validation croisée stratifiée sur le train.

    Args:
        X_train (pd.DataFrame): variables d'entraînement.
        y_train (pd.Series): cible d'entraînement.
        model_types (list[str] | None): modèles à comparer (tous par défaut).
        n_splits (int): nombre de plis.

    Returns:
        pd.DataFrame: Accuracy, F1, AUC (moyenne en CV) et écart-type de l'AUC,
        triés par AUC décroissante.
    """
    model_types = model_types or MODEL_CHOICES
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name in model_types:
        try:
            get_model(name)
        except ImportError as exc:
            print(f"[compare_models] '{name}' ignoré : {exc}")
            continue
        scores = cross_validate(
            get_model(name),
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc"],
            n_jobs=-1,
        )
        rows.append(
            {
                "Modèle": name,
                "Accuracy (CV)": scores["test_accuracy"].mean(),
                "F1 (CV)": scores["test_f1"].mean(),
                "AUC (CV)": scores["test_roc_auc"].mean(),
                "Écart-type AUC": scores["test_roc_auc"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("Modèle").sort_values("AUC (CV)", ascending=False).round(3)


# --------------------------------------------------------------------------
# 4. SAUVEGARDE / CHARGEMENT
# --------------------------------------------------------------------------
def save_model(model, path="model.joblib"):
    """Sauvegarde un modèle entraîné avec joblib.

    Args:
        model: modèle entraîné.
        path (str): chemin du fichier de sortie.

    Returns:
        str: chemin du fichier créé.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Modèle sauvegardé : {path}")
    return path


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier .joblib.

    Returns:
        Le modèle chargé.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}. Lancez d'abord --train.")
    model = joblib.load(path)
    print(f"[load_model] Modèle chargé : {path}")
    return model
