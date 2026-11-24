"""
main.py
=======
Point d'entrée du pipeline : exécute les étapes de model_pipeline.py via la CLI.

Exemples :
    python main.py --prepare
    python main.py --train --model xgboost --save model.joblib
    python main.py --train --model xgboost --tune --save model.joblib
    python main.py --evaluate --load model.joblib
    python main.py --compare
    python main.py --all                      # prepare + train + save + evaluate
"""

import argparse

from model_pipeline import (
    MODEL_CHOICES,
    baseline_accuracy,
    compare_models,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML - Adult Census Income")

    parser.add_argument("--data", default="adult.data.csv", help="Chemin du fichier CSV")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=MODEL_CHOICES,
        help="Algorithme à entraîner (défaut : xgboost)",
    )
    parser.add_argument(
        "--model-path", default="model.joblib", help="Fichier de sauvegarde / chargement du modèle"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil de décision pour l'évaluation (défaut : 0.5)",
    )

    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner (et sauvegarder) le modèle")
    parser.add_argument(
        "--tune", action="store_true", help="Optimiser XGBoost (RandomizedSearchCV)"
    )
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle sur le test")
    parser.add_argument("--compare", action="store_true", help="Comparer les 5 modèles (CV)")
    parser.add_argument("--all", action="store_true", help="prepare + train + save + evaluate")
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.prepare, args.train, args.evaluate, args.compare, args.all]):
        print("Aucune action demandée. Utilisez --help pour voir les options.")
        return

    # 1) Préparation des données (nécessaire pour toutes les autres étapes)
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.prepare and not (args.train or args.evaluate or args.compare or args.all):
        print("Données prêtes.")
        return

    # 2) Comparaison des modèles
    if args.compare:
        print(compare_models(X_train, y_train))

    # 3) Entraînement + sauvegarde
    model = None
    if args.train or args.all:
        model = train_model(X_train, y_train, model_type=args.model, tune=args.tune)
        save_model(model, args.model_path)

    # 4) Évaluation
    if args.evaluate or args.all:
        if model is None:
            model = load_model(args.model_path)
        print(f"[main] Baseline (classe majoritaire) : {baseline_accuracy(y_test):.3f}")
        evaluate_model(model, X_test, y_test, threshold=args.threshold)


if __name__ == "__main__":
    main()
