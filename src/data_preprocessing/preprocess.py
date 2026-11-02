# preprocess.py   - cleaning du CSV consolidé (data/raw/yield_df.csv) pour produire data/consolidated_yield.csv)
#
# Usage : python src/preprocess.py 
#         python src/preprocess.py config.yaml  (tout fichier de config alternatif est accepté)

import sys, yaml
import numpy as np, pandas as pd
from pathlib import Path

def preprocess(cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(cfg["paths"]["raw"]).drop(columns="Unnamed: 0", errors="ignore")
    raw.columns = ["area", "item", "year", "yield_hg_ha", "rain_mm", "pesticides_t", "temp_c"]

    # 1 ligne par (pays, culture, année) : moyenne des temp en double (artefact du merge d'origine)
    df = (raw.groupby(cfg["dedup_keys"], as_index=False)
             .agg(yield_hg_ha=("yield_hg_ha", "first"),
                  rain_mm=("rain_mm", "first"),
                  pesticides_t=("pesticides_t", "first"),
                  temp_c=("temp_c", "mean"))
             .dropna())

    if len(df) != cfg["expected_rows"]:
        raise ValueError(f"{len(df)} lignes au lieu de {cfg['expected_rows']}")

    df["yield_t_ha"] = df["yield_hg_ha"] / 10_000
    df["log_yield"] = np.log1p(df["yield_t_ha"])
    df["log_pest"] = np.log1p(df["pesticides_t"])
    return df.drop(columns="yield_hg_ha")

def add_pesticides_kg_ha(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    add = pd.read_csv(cfg["paths"]["additional"])
    g = add[["Area", "Year", "Value"]].rename(columns={"Area": "area", "Year": "year", "Value": "pest_kg_ha"})
    g["area"] = g["area"].replace({
        "Netherlands (Kingdom of the)": "Netherlands",
        "Türkiye": "Turkey",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    })
    df = df.merge(g, on=["area", "year"], how="left",validate="m:1")
    assert df["pest_kg_ha"].notna().all() and len(df) == cfg["expected_rows"]
    return df

def add_item_fr(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne 'item_fr' en traduisant 'item' via le dictionnaire dans la config."""
    idx = df.columns.get_loc("item") + 1
    df.insert(idx, "item_fr", df["item"].map(cfg["item_fr"]))
    return df

if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml", encoding='utf-8-sig'))
    out = preprocess(cfg)
    out = add_pesticides_kg_ha(cfg,out)
    out = add_item_fr(cfg, out)

    # Réorganisation finale des colonnes
    out = out[[
        "area", "item", "item_fr", "year", 
        "rain_mm", "pesticides_t", "log_pest", "pest_kg_ha", 
        "temp_c", "yield_t_ha", "log_yield"
    ]]

    Path(cfg["paths"]["clean"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["paths"]["clean"], index=False, encoding='utf-8-sig')
    
    print("OK", out.shape)