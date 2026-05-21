import json

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from src.config import ROOT_DIR, load_config
from src.features import build_processed_datasets
from src.nn_model import Classifier

def make_models(config):
    '''
    Loading models from config
    '''
    seed = config["project"]["seed"]
    catboost_params = config["models"]["catboost"]
    random_forest_params = config["models"]["random_forest"]

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "random_forest": RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **random_forest_params,
        ),
        "catboost": CatBoostClassifier(
            random_seed=seed,
            verbose=0,
            loss_function="Logloss",
            eval_metric="Accuracy",
            **catboost_params,
        ),
        "neural_network": Classifier(
            hidden_dims=[128, 64, 32],
            dropout=0.3,
            learning_rate=1e-3,
            batch_size=32,
            epochs=300,
            patience=20,
            device="cpu",
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            random_state=seed,
            verbose=-1,
            **config["models"].get("lightgbm", {}),
),
    }


def score_model(y_true, predictions, probabilities):
    '''
    Scoring models
    '''
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def cross_validate_model(model, x, y, config):
    cv = StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=True,
        random_state=config["project"]["seed"],
    )

    fold_scores = []

    for train_idx, valid_idx in cv.split(x, y):
        fold_model = model.__class__(**model.get_params())

        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions = fold_model.predict(x.iloc[valid_idx]).astype(int)
        probabilities = fold_model.predict_proba(x.iloc[valid_idx])[:, 1]

        scores = score_model(y.iloc[valid_idx], predictions, probabilities)
        fold_scores.append(scores)

    scores_df = pd.DataFrame(fold_scores)

    return {
        "fold_metrics": fold_scores,
        "summary": {
            metric: {
                "mean": scores_df[metric].mean(),
                "std": scores_df[metric].std(ddof=0),
            }
            for metric in scores_df.columns
        },
    }


def compare_models(x, y, config):
    results = {}

    for model_name, model in make_models(config).items():
        print(f"Training {model_name}...")
        results[model_name] = cross_validate_model(model, x, y, config)

    return results


def find_best_model(results):
    return max(
        results,
        key=lambda model_name: results[model_name]["summary"]["accuracy"]["mean"],
    )


def train(train_df=None, feature_info=None):
    config = load_config()

    if train_df is None or feature_info is None:
        train_df, _, feature_info = build_processed_datasets()

    x = train_df.drop(columns="Survived")
    y = train_df["Survived"]

    metrics = compare_models(x, y, config)
    best_model_name = find_best_model(metrics)

    model = make_models(config)[best_model_name]
    model.fit(x, y)

    models_dir = ROOT_DIR / config["paths"]["models_dir"]
    reports_dir = ROOT_DIR / config["paths"]["reports_dir"]
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    artifact = {
        "model": model,
        "model_name": best_model_name,
        "feature_info": feature_info,
    }

    joblib.dump(artifact, models_dir / config["paths"]["model_file"])

    with open(reports_dir / config["paths"]["metrics_file"], "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_model_name,
                "models": metrics,
            },
            file,
            indent=2,
        )

    return {
        "best_model": best_model_name,
        "models": metrics,
    }


if __name__ == "__main__":
    result = train()
    best_model = result["best_model"]
    print(f"\nBest model: {best_model}")
    print(json.dumps(result["models"][best_model]["summary"], indent=2))
