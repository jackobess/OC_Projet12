"""common.py : briques partagées par train.py et evaluate.py (config, données, split, pipeline sklearn, métriques, MLflow)."""
import hashlib
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")

log = logging.getLogger("pipeline")

PREFIX = "regressor__model__"     # préfixe des hyperparamètres dans le pipeline (TransformedTargetRegressor > Pipeline > model)
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

DEFAULT_MODELING = {
    "target": "yield_t_ha",
    "features": ["item", "rain_mm", "temp_c", "pest_kg_ha"],
    "base_features": ["item", "rain_mm", "temp_c"],
    "categorical": ["item", "area"],
    "test_size": 0.2, "cv_folds": 5, "n_iter": 20, "n_jobs": -1,
}
DEFAULT_MLFLOW = {
    "tracking_uri": "sqlite:///mlflow.db",
    "exp_models": "yield-model-selection",
    "exp_ablation": "yield-ablation-variables",
    "registered_model": "yield_model",
}
DEFAULT_GATE = {"max_ratio_vs_baseline": 1.0, "min_reco_vs_random": 1.0}


# ----------------------------------------------------------------------------- config
@dataclass
class Ctx:
    cfg: dict
    root: Path
    seed: int
    target: str
    feats: list
    base: list
    cat: list
    test_size: float
    cv_folds: int
    n_iter: int
    n_jobs: int
    search_spaces: dict
    tracking_uri: str
    exp_models: str
    exp_ablation: str
    registered_model: str
    gate: dict
    clean_path: Path
    candidate_dir: Path
    model_path: Path


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _resolve(root: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_ctx(config_path="config.yaml") -> Ctx:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    root = config_path.parent
    m = {**DEFAULT_MODELING, **cfg.get("modeling", {})}
    ml = {**DEFAULT_MLFLOW, **m.get("mlflow", {})}
    if not m.get("search_spaces"):
        raise ValueError("config.yaml : section modeling.search_spaces manquante")
    paths = cfg["paths"]
    return Ctx(
        cfg=cfg, root=root, seed=cfg["seed"], target=m["target"], feats=m["features"], base=m["base_features"],
        cat=m["categorical"], test_size=m["test_size"], cv_folds=m["cv_folds"], n_iter=m["n_iter"],
        n_jobs=m["n_jobs"], search_spaces=m["search_spaces"], tracking_uri=ml["tracking_uri"],
        exp_models=ml["exp_models"], exp_ablation=ml["exp_ablation"], registered_model=ml["registered_model"],
        gate={**DEFAULT_GATE, **cfg.get("quality_gate", {})},
        clean_path=_resolve(root, paths["clean"]),
        candidate_dir=_resolve(root, paths.get("candidate_dir", "models/candidate")),
        model_path=_resolve(root, paths.get("model", "models/yield_model.joblib")),
    )


def build_search_space(spec: dict) -> dict:
    """YAML -> espace de recherche sklearn. Liste = choix discret ; {dist: loguniform, low, high} = continu."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, dict):
            if v.get("dist") != "loguniform":
                raise ValueError(f"search_spaces.{k} : seule dist=loguniform est gérée")
            out[PREFIX + k] = loguniform(v["low"], v["high"])
        else:
            out[PREFIX + k] = v
    return out


def lib_versions() -> dict:
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "mlflow": mlflow.__version__}


# ----------------------------------------------------------------------------- données
def sha256_short(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_data(ctx: Ctx) -> pd.DataFrame:
    if not ctx.clean_path.exists():
        raise FileNotFoundError(f"dataset consolidé introuvable : {ctx.clean_path} (lance preprocess.py d'abord)")
    df = pd.read_csv(ctx.clean_path)
    needed = set(ctx.feats) | {ctx.target, "area", "year", "log_pest"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"colonnes absentes du CSV consolidé : {missing}")
    if df[sorted(needed)].isna().any().any():
        raise ValueError("valeurs manquantes dans le CSV consolidé")
    if df.duplicated(ctx.cfg.get("dedup_keys", ["area", "item", "year"])).any():
        raise ValueError("doublons dans le CSV consolidé")
    if not (df.groupby("area")["rain_mm"].nunique() == 1).all():
        log.warning("rain_mm n'est plus constante par pays : la justification du split par pays est à revoir")
    return df


def split_by_area(df: pd.DataFrame, ctx: Ctx):
    """Split par pays entiers (rain_mm identifie le pays : un split aléatoire gonflerait le score)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=ctx.test_size, random_state=ctx.seed)
    tr_idx, te_idx = next(gss.split(df, groups=df["area"]))
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    if not set(train["area"]).isdisjoint(test["area"]):
        raise ValueError("fuite : un pays est à la fois en train et en test")
    log.info("train : %d lignes / %d pays | test : %d lignes / %d pays",
             len(train), train["area"].nunique(), len(test), test["area"].nunique())
    return train, test


def data_tags(ctx: Ctx, df: pd.DataFrame) -> dict:
    """Tags MLflow qui rendent chaque run traçable à ses données."""
    return {"data_sha": sha256_short(ctx.clean_path), "n_rows": str(len(df)),
            "n_areas": str(df["area"].nunique()), "years": f"{df['year'].min()}-{df['year'].max()}"}


# ----------------------------------------------------------------------------- modèles
def build_model(name: str, seed: int):
    models = {
        "dummy_mean":        lambda: DummyRegressor(strategy="mean"),
        "item_mean":         lambda: LinearRegression(),   # sur 'item' seul = rendement moyen par culture
        "ridge":             lambda: Ridge(alpha=1.0),
        "random_forest":     lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=seed),
        "xgboost":           lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0),
        "lightgbm":          lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                                   subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                                                   random_state=seed, n_jobs=1, verbose=-1),
    }
    if name not in models:
        raise ValueError(f"modèle inconnu '{name}' (disponibles : {sorted(models)})")
    return models[name]()


def make_pipe(name: str, feats: list, ctx: Ctx):
    """Preprocessing + modèle. Cible entraînée en log1p, prédictions rendues en t/ha."""
    cat = [c for c in feats if c in ctx.cat]
    num = [c for c in feats if c not in ctx.cat]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    pipe = Pipeline([("prep", prep), ("model", build_model(name, ctx.seed))])
    return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


# ----------------------------------------------------------------------------- métriques
def reg_metrics(y, p, prefix: str) -> dict:
    return {f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, p))),
            f"{prefix}_mae": float(mean_absolute_error(y, p)),
            f"{prefix}_r2": float(r2_score(y, p))}


def reco_top1(pred, data: pd.DataFrame, item_mean: pd.Series, target: str):
    """Pour chaque (pays, année) du test : la culture classée 1re par le modèle (écart vs norme de la culture)
    est-elle vraiment la meilleure ? Retourne (taux de bonne reco, taux attendu au hasard)."""
    d = data[["area", "year", "item", target]].copy()
    d["real_rel"] = d[target] / d["item"].map(item_mean)
    d["pred_rel"] = pred / d["item"].map(item_mean)
    hits, rnd = [], []
    for _, g in d.groupby(["area", "year"]):
        if len(g) < 3:
            continue
        hits.append(g["real_rel"].idxmax() == g["pred_rel"].idxmax())
        rnd.append(1 / len(g))
    return float(np.mean(hits)), float(np.mean(rnd))


def holdout_metrics(est, train, test, feats, ctx: Ctx, item_mean) -> dict:
    """Métriques train/test, écart (overfit) et qualité de la recommandation."""
    p_tr, p_te = est.predict(train[feats]), est.predict(test[feats])
    m = reg_metrics(train[ctx.target], p_tr, "train") | reg_metrics(test[ctx.target], p_te, "test")
    m["gap_rmse"] = m["test_rmse"] - m["train_rmse"]     # > 0 : le modèle fait pire sur des pays inconnus
    m["gap_r2"] = m["train_r2"] - m["test_r2"]
    m["reco_top1"], m["reco_top1_random"] = reco_top1(p_te, test, item_mean, ctx.target)
    return m


# ----------------------------------------------------------------------------- MLflow
def setup_mlflow(ctx: Ctx) -> None:
    """URI depuis MLFLOW_TRACKING_URI (Docker/CI) sinon depuis la config ; sqlite relatif -> relatif à la racine."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or ctx.tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        uri = prefix + (ctx.root / uri[len(prefix):]).as_posix()
    mlflow.set_tracking_uri(uri)
    log.info("MLflow tracking : %s", uri)


def set_exp(ctx: Ctx, name: str) -> None:
    """Artefacts dans <MLFLOW_ARTIFACT_ROOT ou racine/mlruns>/<expérience>."""
    if mlflow.get_experiment_by_name(name) is None:
        art_root = Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", ctx.root / "mlruns"))
        mlflow.create_experiment(name, artifact_location=(art_root / name).resolve().as_uri())
    mlflow.set_experiment(name)
