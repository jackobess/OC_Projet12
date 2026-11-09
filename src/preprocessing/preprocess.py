"""preprocess.py : brut (yield_df.csv + pesticides_par_ha.csv) -> data/consolidated_yield.csv

Usage (depuis n'importe où) :
    python src/preprocessing/preprocess.py                  # config.yaml (dossier courant)
    python src/preprocessing/preprocess.py config.yaml --raw data/new/yield_df.csv

Étapes de validation (dans l'ordre) :
    1. colonnes requises            -> erreur
    2. NaN                          -> warning par colonne, lignes supprimées
    3. cultures absentes de item_fr -> erreur (liste à ajouter dans config.yaml)
    4. valeurs hors plages          -> warning par colonne, lignes supprimées
    5. trop de lignes perdues       -> erreur (validation.max_dropped_pct)
    6. nb de lignes finales         -> erreur (validation.min_rows)
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("preprocess")

# Valeurs par défaut : surchargeables dans config.yaml (clés raw_columns, validation, area_aliases)
DEFAULT_RAW_COLUMNS = {
    "Area": "area", "Item": "item", "Year": "year", "hg/ha_yield": "yield_hg_ha",
    "average_rain_fall_mm_per_year": "rain_mm", "pesticides_tonnes": "pesticides_t", "avg_temp": "temp_c",
}
DEFAULT_VALIDATION = {
    "ranges": {"temp_c": [-30, 45], "rain_mm": [0, 6000], "yield_t_ha": [0, 150]},
    "min_rows": 5000,
    "max_dropped_pct": 10,
}
DEFAULT_AREA_ALIASES = {
    "Netherlands (Kingdom of the)": "Netherlands",
    "Türkiye": "Turkey",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
}
INTERNAL_COLUMNS = {"area", "item", "year", "yield_hg_ha", "rain_mm", "pesticides_t", "temp_c"}
NUMERIC = ["year", "yield_hg_ha", "rain_mm", "pesticides_t", "temp_c"]
FINAL_COLUMNS = ["area", "item", "item_fr", "year", "rain_mm", "pesticides_t", "log_pest",
                 "pest_kg_ha", "temp_c", "yield_t_ha", "log_yield"]


def clean_raw(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validation + nettoyage du brut. Retourne les colonnes internes + yield_t_ha."""
    mapping = cfg.get("raw_columns", DEFAULT_RAW_COLUMNS)
    v = {**DEFAULT_VALIDATION, **cfg.get("validation", {})}
    n0 = len(raw)

    if set(mapping.values()) != INTERNAL_COLUMNS:
        raise ValueError(f"raw_columns doit produire exactement {sorted(INTERNAL_COLUMNS)}, "
                         f"obtenu {sorted(mapping.values())}")

    # 1. colonnes requises
    missing = sorted(set(mapping) - set(raw.columns))
    if missing:
        raise ValueError(f"colonnes absentes du fichier brut : {missing} (présentes : {list(raw.columns)})")
    df = raw[list(mapping)].rename(columns=mapping).copy()
    for c in NUMERIC:                                   # texte parasite -> NaN (compté à l'étape suivante)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2. NaN
    nan_counts = df.isna().sum()
    for col, n in nan_counts[nan_counts > 0].items():
        log.warning("%d NaN dans '%s'", n, col)
    df = df.dropna()
    df["year"] = df["year"].astype(int)

    # 3. cultures inconnues
    unknown = sorted(set(df["item"]) - set(cfg["item_fr"]))
    if unknown:
        raise ValueError(f"cultures à ajouter et traduire dans config.yaml > item_fr : {unknown}")

    # 4. valeurs hors plages (rendement contrôlé en t/ha, après conversion d'unité)
    df["yield_t_ha"] = df["yield_hg_ha"] / 10_000
    for col, (lo, hi) in v["ranges"].items():
        if col not in df.columns:
            raise ValueError(f"validation.ranges : colonne inconnue '{col}'")
        bad = ~df[col].between(lo, hi)
        if bad.any():
            log.warning("%d valeurs de '%s' hors [%s, %s] : lignes supprimées", bad.sum(), col, lo, hi)
            df = df[~bad]

    # 5. pertes trop importantes (mesurées sur le brut, avant dédoublonnage)
    dropped = 100 * (n0 - len(df)) / n0
    log.info("nettoyage : %d -> %d lignes (%.1f %% supprimées)", n0, len(df), dropped)
    if dropped > v["max_dropped_pct"]:
        raise ValueError(f"{dropped:.1f} % de lignes supprimées (max autorisé : {v['max_dropped_pct']} %)")

    return df.drop(columns="yield_hg_ha")


def preprocess(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """1 ligne par (area, item, year) + variables dérivées."""
    df = clean_raw(raw, cfg)
    keys = cfg.get("dedup_keys", ["area", "item", "year"])

    # le brut vient d'un merge : seules les temp sont en double, on vérifie que le reste est constant
    grp = df.groupby(keys)
    varying = (grp[["yield_t_ha", "rain_mm", "pesticides_t"]].nunique() > 1).sum()
    for col, n in varying[varying > 0].items():
        log.warning("%d groupes avec plusieurs valeurs de '%s' : première valeur conservée", n, col)

    df = (df.groupby(keys, as_index=False)
            .agg(yield_t_ha=("yield_t_ha", "first"),
                 rain_mm=("rain_mm", "first"),
                 pesticides_t=("pesticides_t", "first"),
                 temp_c=("temp_c", "mean")))

    df["log_yield"] = np.log1p(df["yield_t_ha"])
    df["log_pest"] = np.log1p(df["pesticides_t"])
    if df.isna().any().any():
        raise ValueError(f"NaN inattendus après feature engineering : {df.isna().sum()[lambda s: s > 0].to_dict()}")
    return df


def add_pesticides_kg_ha(cfg: dict, df: pd.DataFrame, path: Path) -> pd.DataFrame:
    add = pd.read_csv(path, encoding="utf-8-sig")
    need = {"Area", "Year", "Value"}
    if need - set(add.columns):
        raise ValueError(f"{path.name} : colonnes absentes {sorted(need - set(add.columns))}")

    g = add[["Area", "Year", "Value"]].rename(columns={"Area": "area", "Year": "year", "Value": "pest_kg_ha"})
    g["area"] = g["area"].replace(cfg.get("area_aliases", DEFAULT_AREA_ALIASES))

    df = df.merge(g, on=["area", "year"], how="left", validate="m:1")   # MergeError si doublons côté pesticides

    miss = df["pest_kg_ha"].isna()
    if miss.any():
        per_area = df[miss].groupby("area").size()
        total = df.groupby("area").size()
        fully_missing = sorted(a for a in per_area.index if per_area[a] == total[a])
        if fully_missing:
            raise ValueError(f"pays sans aucune donnée pesticides/ha (nom différent ? voir config.yaml > "
                             f"area_aliases) : {fully_missing}")
        log.warning("%d lignes sans pesticides/ha (années manquantes) supprimées : %s",
                    miss.sum(), per_area.to_dict())
        df = df[~miss]
    return df


def add_item_fr(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(df.columns.get_loc("item") + 1, "item_fr", df["item"].map(cfg["item_fr"]))
    return df


def check_min_rows(df: pd.DataFrame, cfg: dict) -> None:
    min_rows = {**DEFAULT_VALIDATION, **cfg.get("validation", {})}["min_rows"]
    if len(df) < min_rows:
        raise ValueError(f"{len(df)} lignes finales < minimum requis ({min_rows})")


def _resolve(root: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def run(config_path="config.yaml", raw=None, additional=None, out=None) -> pd.DataFrame:
    """Point d'entrée importable (run_pipeline.py). Les chemins relatifs sont relatifs au dossier du config."""
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    root = config_path.parent

    raw_p = Path(raw).resolve() if raw else _resolve(root, cfg["paths"]["raw"])
    add_p = Path(additional).resolve() if additional else _resolve(root, cfg["paths"]["additional"])
    out_p = Path(out).resolve() if out else _resolve(root, cfg["paths"]["clean"])
    for p in (raw_p, add_p):
        if not p.exists():
            raise FileNotFoundError(f"fichier introuvable : {p}")

    log.info("brut : %s | pesticides/ha : %s", raw_p, add_p)
    df = preprocess(pd.read_csv(raw_p, encoding="utf-8-sig"), cfg)
    df = add_pesticides_kg_ha(cfg, df, add_p)
    df = add_item_fr(cfg, df)[FINAL_COLUMNS]
    check_min_rows(df, cfg)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False, encoding="utf-8")
    log.info("OK %s -> %s", df.shape, out_p)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--raw", help="surcharge paths.raw")
    ap.add_argument("--additional", help="surcharge paths.additional")
    ap.add_argument("--out", help="surcharge paths.clean")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    try:
        run(a.config, a.raw, a.additional, a.out)
    except (ValueError, FileNotFoundError, pd.errors.MergeError) as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
