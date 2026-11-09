"""train.py : entrainement & comparaison de modèles + optimisation d'hyperparamètres + sélection (tout loggé dans MLflow).

Entrée  : data/consolidated_yield.csv (sortie de preprocess.py) + config.yaml
Sorties : <candidate_dir>/estimator.joblib    meilleur modèle entraîné sur le train seul (pour evaluate.py)
          <candidate_dir>/bundle.joblib       modèle de prod (ré-entraîné sur tout le dataset) + métadonnées pour l'API
          <candidate_dir>/train_report.json   résultats, meilleurs hyperparamètres, empreinte des données
Le bundle n'est promu en models/yield_model.joblib qu'après le contrôle qualité d'evaluate.py.

Usage : python src/preprocessing/train.py [config.yaml]
"""
import argparse
import json
import logging
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate

from common import (PREFIX, SCORING, build_search_space, data_tags, holdout_metrics, lib_versions, load_ctx,
                    load_data, make_pipe, reg_metrics, set_exp, setup_logging, setup_mlflow, split_by_area)

log = logging.getLogger("train")

REPORT_COLS = ["cv_rmse", "cv_mae", "cv_r2", "test_rmse", "test_mae", "test_r2", "gap_rmse", "reco_top1", "reco_top1_random"]


def comparison_runs(ctx):
    """(nom du run, modèle, variables, famille, note MLflow)."""
    return [
        ("baseline_dummy_mean", "dummy_mean", ["item"], "baseline",
         "Baseline minimale : prédit toujours la moyenne globale. Tout modèle doit faire mieux."),
        ("baseline_item_mean", "item_mean", ["item"], "baseline",
         "Baseline métier : rendement moyen par culture, sans pluie ni température. Mesure ce que le climat apporte en plus de la culture."),
        ("ridge", "ridge", ctx.feats, "linear", "Linéaire régularisé : référence interprétable, sans interactions."),
        ("random_forest", "random_forest", ctx.feats, "tree_ensemble", "RandomForest, hyperparamètres modérés (200 arbres, min_samples_leaf=5)."),
        ("gradient_boosting", "gradient_boosting", ctx.feats, "tree_ensemble", "GradientBoosting, hyperparamètres par défaut."),
        ("xgboost", "xgboost", ctx.feats, "tree_ensemble", "XGBoost, hyperparamètres modérés (300 arbres, learning_rate=0.05, max_depth=4, subsample=0.8)."),
        ("lightgbm", "lightgbm", ctx.feats, "tree_ensemble", "LightGBM, hyperparamètres modérés (300 arbres, learning_rate=0.05, 15 feuilles, subsample=0.8)."),
    ]


def fit_eval_log(ctx, train, test, item_mean, tags, run_name, model_name, feats, family, note) -> dict:
    """CV groupée par pays sur le train + fit complet + métriques train/test + gap, loggés dans MLflow."""
    pipe = make_pipe(model_name, feats, ctx)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({**tags, "family": family, "stage": "model_selection", "mlflow.note.content": note})
        mlflow.log_params({"features": ",".join(feats), "split": "GroupShuffleSplit(area)",
                           "cv": f"GroupKFold({ctx.cv_folds})", "seed": ctx.seed, "target": f"log1p({ctx.target})"})
        mlflow.log_params({f"model__{k}": v for k, v in pipe.regressor.named_steps["model"].get_params().items()})

        cv = cross_validate(pipe, train[feats], train[ctx.target], groups=train["area"],
                            cv=GroupKFold(ctx.cv_folds), scoring=SCORING, n_jobs=ctx.n_jobs)
        m = {"cv_rmse": -cv["test_rmse"].mean(), "cv_rmse_std": cv["test_rmse"].std(),
             "cv_mae": -cv["test_mae"].mean(), "cv_r2": cv["test_r2"].mean()}
        pipe.fit(train[feats], train[ctx.target])
        m |= holdout_metrics(pipe, train, test, feats, ctx, item_mean)
        mlflow.log_metrics(m)
    log.info("%-22s cv_rmse %.3f | test_rmse %.3f | test_r2 %.3f", run_name, m["cv_rmse"], m["test_rmse"], m["test_r2"])
    return m


def tune(ctx, train, test, item_mean, tags, name, space):
    """RandomizedSearch (GroupKFold par pays) ; un run parent + un run enfant par candidat."""
    srch = RandomizedSearchCV(make_pipe(name, ctx.feats, ctx), build_search_space(space), n_iter=ctx.n_iter,
                              scoring=SCORING, refit="rmse", cv=GroupKFold(ctx.cv_folds), n_jobs=ctx.n_jobs,
                              random_state=ctx.seed, return_train_score=True)
    with mlflow.start_run(run_name=f"tuning_{name}"):
        mlflow.set_tags({**tags, "stage": "tuning", "family": "search", "mlflow.note.content":
                         f"RandomizedSearch ({ctx.n_iter} candidats, GroupKFold {ctx.cv_folds} par pays) sur {name}. Un run enfant par candidat."})
        mlflow.log_params({"model": name, "n_iter": ctx.n_iter, "cv": f"GroupKFold({ctx.cv_folds})",
                           "seed": ctx.seed, "features": ",".join(ctx.feats)})
        srch.fit(train[ctx.feats], train[ctx.target], groups=train["area"])

        cvr = srch.cv_results_
        for i, params in enumerate(cvr["params"]):
            with mlflow.start_run(run_name=f"{name}_candidat_{i:02d}", nested=True):
                mlflow.log_params({k.replace(PREFIX, ""): v for k, v in params.items()})
                mlflow.log_metrics({"cv_rmse": -cvr["mean_test_rmse"][i], "cv_rmse_std": cvr["std_test_rmse"][i],
                                    "cv_mae": -cvr["mean_test_mae"][i], "cv_r2": cvr["mean_test_r2"][i],
                                    "cv_train_rmse": -cvr["mean_train_rmse"][i]})
                mlflow.set_tag("rank_cv_rmse", int(cvr["rank_test_rmse"][i]))

        est, k = srch.best_estimator_, srch.best_index_      # déjà ré-entraîné sur tout le train
        best_params = {kk.replace(PREFIX, ""): v for kk, v in srch.best_params_.items()}
        mlflow.log_params({f"best__{kk}": v for kk, v in best_params.items()})
        m = {"cv_rmse": -cvr["mean_test_rmse"][k], "cv_rmse_std": cvr["std_test_rmse"][k],
             "cv_mae": -cvr["mean_test_mae"][k], "cv_r2": cvr["mean_test_r2"][k]}
        m |= holdout_metrics(est, train, test, ctx.feats, ctx, item_mean)
        mlflow.log_metrics(m)
    log.info("%-18s cv_rmse %.3f | test_rmse %.3f | params %s", name, m["cv_rmse"], m["test_rmse"],
             {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in best_params.items()})
    return est, best_params, m


def build_bundle(ctx, df, prod_model, name, best_params) -> dict:
    """Tout ce dont l'API a besoin, calculé sur l'ensemble des données."""
    ranged = [c for c in ctx.feats if c != "item"]
    return {
        "model": prod_model,
        "features": ctx.feats,
        "items": sorted(df["item"].unique()),
        "item_mean_t_ha": df.groupby("item")[ctx.target].mean().to_dict(),     # "norme" de chaque culture
        "model_name": name,
        "best_params": best_params,
        "sklearn_version": lib_versions()["sklearn"],
        "versions": lib_versions(),
        "seed": ctx.seed,
        "data_sha": data_tags(ctx, df)["data_sha"],
        "input_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in ranged},
        "item_ranges": {it: {c: {"min": float(g[c].min()), "max": float(g[c].max())} for c in ranged}
                        for it, g in df.groupby("item")},
    }


def run(config_path="config.yaml") -> dict:
    ctx = load_ctx(config_path)
    np.random.seed(ctx.seed)
    setup_mlflow(ctx)
    df = load_data(ctx)
    train, test = split_by_area(df, ctx)
    item_mean = train.groupby("item")[ctx.target].mean()       # norme de chaque culture (train uniquement)
    tags = data_tags(ctx, df)
    set_exp(ctx, ctx.exp_models)

    # 1. baselines + comparaison de modèles (hyperparamètres par défaut / modérés)
    results = []
    for run_name, model_name, feats, family, note in comparison_runs(ctx):
        m = fit_eval_log(ctx, train, test, item_mean, tags, run_name, model_name, feats, family, note)
        results.append({"run": run_name, "model": model_name, **m})

    # 2. optimisation des modèles à arbres (capables de capter les interactions culture x climat)
    tuned = {}
    for name, space in ctx.search_spaces.items():
        est, best_params, m = tune(ctx, train, test, item_mean, tags, name, space)
        tuned[name] = {"est": est, "params": best_params, "metrics": m}
        results.append({"run": f"tuned_{name}", "model": name, **m})

    # 3. sélection sur cv_rmse (le test ne sert qu'à vérifier la généralisation)
    best_name = min(tuned, key=lambda n: tuned[n]["metrics"]["cv_rmse"])
    best = tuned[best_name]
    log.info("=> modèle retenu : %s (cv_rmse %.3f)", best_name, best["metrics"]["cv_rmse"])
    log.info("\n%s", pd.DataFrame(results).set_index("run")[REPORT_COLS].round(3).to_string())

    # 4. modèle de prod : mêmes hyperparamètres, ré-entraîné sur TOUTES les données (le test a déjà servi à l'évaluer)
    prod_model = clone(best["est"]).fit(df[ctx.feats], df[ctx.target])
    bundle = build_bundle(ctx, df, prod_model, best_name, best["params"])

    ctx.candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["est"], ctx.candidate_dir / "estimator.joblib")
    joblib.dump(bundle, ctx.candidate_dir / "bundle.joblib")
    check = joblib.load(ctx.candidate_dir / "bundle.joblib")           # contrôle : rechargement identique
    if not np.allclose(check["model"].predict(df[ctx.feats].head(50)), prod_model.predict(df[ctx.feats].head(50))):
        raise RuntimeError("le bundle rechargé ne donne pas les mêmes prédictions")

    report = {"best_name": best_name, "best_params": best["params"], "tuning_metrics": best["metrics"],
              "results": results, "features": ctx.feats, "data_sha": tags["data_sha"],
              "test_areas": sorted(test["area"].unique()), "versions": lib_versions()}
    (ctx.candidate_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    log.info("OK candidat écrit dans %s", ctx.candidate_dir)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config.yaml")
    setup_logging()
    try:
        run(ap.parse_args().config)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
