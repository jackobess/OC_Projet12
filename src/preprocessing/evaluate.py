"""evaluate.py : évaluation du modèle candidat (jeu de test = pays jamais vus), importance des variables,
ablation, contrôle qualité, puis promotion du bundle en models/yield_model.joblib si le contrôle passe.

Entrée  : <candidate_dir>/ (sorties de train.py) + data/consolidated_yield.csv + config.yaml
Sorties : runs MLflow (final_model + ablation), figures, models/yield_model.joblib (si contrôle qualité OK)
Code retour 1 si le contrôle qualité échoue (modèle pas meilleur que la baseline, reco pas meilleure que le hasard).

Usage : python src/preprocessing/evaluate.py [config.yaml]
"""
import argparse
import json
import logging
import shutil
import sys

import joblib
import matplotlib
matplotlib.use("Agg")          # pas d'écran dans Docker / CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from common import (PREFIX, holdout_metrics, load_ctx, load_data, make_pipe, reg_metrics, set_exp, setup_logging,
                    setup_mlflow, sha256_short, split_by_area)

log = logging.getLogger("evaluate")
sns.set_theme(style="whitegrid")

REPORT_COLS = ["cv_rmse", "cv_mae", "cv_r2", "test_rmse", "test_mae", "test_r2", "gap_rmse", "reco_top1", "reco_top1_random"]


class QualityGateError(Exception):
    pass


# ----------------------------------------------------------------------------- figures
def fig_pred_vs_real(test, pred, target):
    items = sorted(test["item"].unique())
    palette = dict(zip(items, sns.color_palette("tab10", n_colors=len(items))))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for it in items:
        m = (test["item"] == it).values
        axes[0].scatter(test.loc[m, target], pred[m], s=8, alpha=.5, color=palette[it], label=it)
    lim = [0, max(test[target].max(), pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("Rendement réel (t/ha)"); axes[0].set_ylabel("Prédit (t/ha)")
    axes[0].set_title("Prédit vs réel (pays jamais vus)")
    axes[0].legend(title="Culture", fontsize=7, markerscale=2, loc="upper left")
    err = pd.DataFrame({"item": test["item"].values, "erreur": pred - test[target].values})
    sns.boxplot(data=err, y="item", x="erreur", hue="item", order=items, palette=palette, ax=axes[1],
                showfliers=False, legend=False)
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("Erreur (t/ha)"); axes[1].set_title("Erreur par culture")
    fig.tight_layout()
    return fig


def fig_by_item(test, pred, target):
    col = "item_fr" if "item_fr" in test.columns else "item"
    d = test.assign(predicted=pred)
    g = sns.FacetGrid(d, col=col, col_wrap=3, sharex=False, sharey=False, height=3, aspect=1.3)
    g.map_dataframe(sns.scatterplot, x=target, y="predicted", s=30, alpha=0.5)

    def diag(*args, **kwargs):
        ax = plt.gca()
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="red", lw=1)

    g.map_dataframe(diag)
    g.set_axis_labels("Rendement réel (t/ha)", "Prédit (t/ha)")
    g.set_titles(col_template="{col_name}")
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle("Prédit vs Réel par culture (pays jamais vus)", fontsize=13)
    return g.figure


def fig_importance(imp: pd.Series, std: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 3))
    imp.plot.barh(ax=ax, xerr=std[imp.index])
    ax.set_xlabel("Hausse du RMSE (t/ha) si la variable est mélangée")
    ax.set_title("Importance par permutation (test)")
    fig.tight_layout()
    return fig


def fig_ablation(abl: pd.DataFrame, order: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, metric in zip(axes, ["test_r2", "test_rmse"]):
        sns.barplot(data=abl, y="variables", x=metric, hue="split", ax=ax, order=order, legend=(ax is axes[0]))
        ax.set_title(metric)
        ax.set_xlim(max(0.0, abl[metric].min() * 0.8), None)
    axes[1].set_ylabel("")
    sns.move_legend(axes[0], "center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- ablation
def ablation(ctx, df, train, test, best_name, best_params):
    """Même modèle (hyperparamètres optimisés), différents jeux de variables, split par pays vs aléatoire."""
    sets = {
        "base (item+pluie+temp)":              ctx.base,
        "retenu (base + pest. kg/ha)":         ctx.feats,
        "retenu + year":                       ctx.feats + ["year"],
        "base + pest. total (t)":              ctx.base + ["log_pest"],
        "retenu + area":                       ctx.feats + ["area"],
        "tout (retenu+area+year+pest. total)": ctx.feats + ["area", "year", "log_pest"],
    }
    rtr, rte = train_test_split(df, test_size=ctx.test_size, random_state=ctx.seed)
    splits = {"par_pays": (train, test), "aleatoire": (rtr, rte)}
    params = {PREFIX + k: v for k, v in best_params.items()}
    rows = []
    set_exp(ctx, ctx.exp_ablation)
    for split_name, (a, b) in splits.items():
        for set_name, feats in sets.items():
            pipe = make_pipe(best_name, feats, ctx).set_params(**params).fit(a[feats], a[ctx.target])
            m = reg_metrics(b[ctx.target], pipe.predict(b[feats]), "test")
            with mlflow.start_run(run_name=f"{split_name} | {set_name}"):
                mlflow.set_tags({"stage": "ablation", "mlflow.note.content":
                                 "Ablation : apport des variables écartées du modèle final, selon le type de split."})
                mlflow.log_params({"model": best_name, "features": ",".join(feats), "split": split_name, "seed": ctx.seed})
                mlflow.log_metrics(m)
            rows.append({"split": split_name, "variables": set_name, **m})
    return pd.DataFrame(rows), list(sets)


# ----------------------------------------------------------------------------- contrôle qualité
def quality_gate(ctx, report, m) -> dict:
    base = next(r for r in report["results"] if r["run"] == "baseline_item_mean")["test_rmse"]
    g = ctx.gate
    checks = {
        f"test_rmse {m['test_rmse']:.3f} <= {g['max_ratio_vs_baseline']} x baseline item_mean {base:.3f}":
            m["test_rmse"] <= base * g["max_ratio_vs_baseline"],
        f"reco_top1 {m['reco_top1']:.3f} >= {g['min_reco_vs_random']} x hasard {m['reco_top1_random']:.3f}":
            m["reco_top1"] >= m["reco_top1_random"] * g["min_reco_vs_random"],
    }
    for label, ok in checks.items():
        log.info("contrôle qualité [%s] %s", "OK" if ok else "ÉCHEC", label)
    return checks


# ----------------------------------------------------------------------------- pipeline
def run(config_path="config.yaml") -> dict:
    ctx = load_ctx(config_path)
    np.random.seed(ctx.seed)
    setup_mlflow(ctx)

    rep_path = ctx.candidate_dir / "train_report.json"
    for p in (rep_path, ctx.candidate_dir / "estimator.joblib", ctx.candidate_dir / "bundle.joblib"):
        if not p.exists():
            raise FileNotFoundError(f"{p} introuvable (lance train.py d'abord)")
    report = json.loads(rep_path.read_text(encoding="utf-8"))

    df = load_data(ctx)
    train, test = split_by_area(df, ctx)
    # cohérence train -> evaluate : mêmes données, même split
    if report["data_sha"] != sha256_short(ctx.clean_path):
        raise ValueError("le CSV consolidé a changé depuis train.py : relance train.py")
    if sorted(test["area"].unique()) != report["test_areas"]:
        raise ValueError("le split test ne correspond plus à celui de train.py (versions de libs différentes ?)")

    est = joblib.load(ctx.candidate_dir / "estimator.joblib")
    bundle = joblib.load(ctx.candidate_dir / "bundle.joblib")
    best_name, best_params = report["best_name"], report["best_params"]
    feats, target = ctx.feats, ctx.target
    item_mean = train.groupby("item")[target].mean()

    # 1. métriques sur le test (cv_* repris du tuning, le reste recalculé)
    m = {k: v for k, v in report["tuning_metrics"].items() if k.startswith("cv_")}
    m |= holdout_metrics(est, train, test, feats, ctx, item_mean)
    if not np.isclose(m["test_rmse"], report["tuning_metrics"]["test_rmse"], rtol=1e-6):
        log.warning("test_rmse recalculé (%.6f) != train.py (%.6f) : résultats non reproductibles ?",
                    m["test_rmse"], report["tuning_metrics"]["test_rmse"])
    log.info("modèle %s | test_rmse %.3f | test_r2 %.3f | gap_rmse %.3f | reco_top1 %.3f (hasard %.3f)",
             best_name, m["test_rmse"], m["test_r2"], m["gap_rmse"], m["reco_top1"], m["reco_top1_random"])

    # 2. figures + importance par permutation
    pred_te = est.predict(test[feats])
    f_eval, f_item = fig_pred_vs_real(test, pred_te, target), fig_by_item(test, pred_te, target)
    pi = permutation_importance(est, test[feats], test[target], scoring="neg_root_mean_squared_error",
                                n_repeats=10, random_state=ctx.seed, n_jobs=ctx.n_jobs)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values()
    f_imp = fig_importance(imp, pd.Series(pi.importances_std, index=feats))
    log.info("importance (hausse de RMSE en t/ha) :\n%s", imp.round(3).to_string())

    # 3. ablation
    abl, order = ablation(ctx, df, train, test, best_name, best_params)
    f_abl = fig_ablation(abl, order)

    # 4. contrôle qualité
    checks = quality_gate(ctx, report, m)
    passed = all(checks.values())

    # 5. run final MLflow (le modèle n'est enregistré dans le registry que si le contrôle passe)
    set_exp(ctx, ctx.exp_models)
    prod_model = bundle["model"]
    with mlflow.start_run(run_name="final_model"):
        mlflow.set_tags({"stage": "final", "quality_gate": "passed" if passed else "failed",
                         "data_sha": report["data_sha"], "mlflow.note.content":
                         f"Modèle final : {best_name} optimisé, ré-entraîné sur toutes les données. "
                         "Métriques = celles du modèle évalué sur le test (pays jamais vus)."})
        mlflow.log_params({"model": best_name, "features": ",".join(feats), "seed": ctx.seed, **best_params})
        mlflow.log_metrics(m)
        mlflow.log_metrics({"importance_" + k: float(v) for k, v in imp.items()})
        mlflow.log_figure(f_eval, "eval_pred_vs_real.png")
        mlflow.log_figure(f_item, "eval_par_culture.png")
        mlflow.log_figure(f_imp, "permutation_importance.png")
        mlflow.log_figure(f_abl, "ablation.png")
        mlflow.log_table(pd.DataFrame(report["results"]).set_index("run")[REPORT_COLS].reset_index(), "comparaison_modeles.json")
        mlflow.log_dict(bundle["item_mean_t_ha"], "item_mean_t_ha.json")
        mlflow.log_dict({"checks": {k: bool(v) for k, v in checks.items()}}, "quality_gate.json")
        sig = infer_signature(df[feats].head(100), prod_model.predict(df[feats].head(100)))
        # cloudpickle : le défaut skops de MLflow >= 3 refuse les types XGBoost/LightGBM
        kw = dict(signature=sig, input_example=df[feats].head(3), serialization_format="cloudpickle")
        if passed:
            kw["registered_model_name"] = ctx.registered_model
        try:
            info = mlflow.sklearn.log_model(prod_model, name="model", **kw)               # MLflow >= 3
        except TypeError:
            info = mlflow.sklearn.log_model(prod_model, artifact_path="model", **kw)      # MLflow 2.x
    plt.close("all")

    if not passed:
        raise QualityGateError("contrôle qualité en échec : le modèle candidat n'est pas promu "
                               f"(models/yield_model.joblib inchangé). Détail : {[k for k, v in checks.items() if not v]}")

    # 6. promotion : le bundle candidat devient le modèle servi par l'API
    ctx.model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ctx.candidate_dir / "bundle.joblib", ctx.model_path)
    version = getattr(info, "registered_model_version", None)
    if version:
        try:
            MlflowClient().set_registered_model_alias(ctx.registered_model, "champion", str(version))
        except Exception as e:                                          # alias non critique
            log.warning("alias 'champion' non posé : %s", e)
    log.info("OK modèle promu -> %s", ctx.model_path)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config.yaml")
    setup_logging()
    try:
        run(ap.parse_args().config)
    except (ValueError, FileNotFoundError, QualityGateError) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
