# API de prédiction / recommandation de rendement agricole (Agritech Answers).
#
# Usage: uvicorn src.api.main:app --reload
# docs interactives : http://localhost:8000/docs

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

# /app/src/api/main.py -> ROOT = /app (idem en local : racine du projet)
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "yield_model.joblib"))
VARS = ("rain_mm", "temp_c", "pest_kg_ha")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH)  # chargé une seule fois au démarrage
    yield


app = FastAPI(
    title="Agritech Answers - Yield API",
    description="Prédiction du rendement d'une culture et recommandation de la meilleure culture.",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------- Schémas ----------
class Context(BaseModel):
    rain_mm: float = Field(..., ge=0, le=6000, description="Pluie annuelle (mm/an)", examples=[1000])
    temp_c: float = Field(..., ge=-10, le=40, description="Température moyenne (°C)", examples=[15])
    pest_kg_ha: float = Field(..., ge=0, le=100, description="Pesticides (kg matière active/ha)", examples=[2])


class PredictRequest(Context):
    item: str = Field(..., description="Culture (nom anglais, cf. GET /getitemlist, insensible à la casse)",
                      examples=["Maize"])


class RecommendRequest(Context):
    sort_by: Literal["ecart_vs_norme", "rendement"] = Field(
        "ecart_vs_norme",
        description="'ecart_vs_norme' : % vs rendement moyen de la culture (défaut) ; 'rendement' : t/ha brut",
    )


class PredictResponse(BaseModel):
    item: str
    predicted_yield_t_ha: float
    norm_t_ha: float
    gap_vs_norm_pct: float
    warnings: list[str] = []


class RecommendItem(BaseModel):
    rank: int
    item: str
    predicted_yield_t_ha: float
    norm_t_ha: float
    gap_vs_norm_pct: float


class ExcludedItem(BaseModel):
    item: str
    reasons: list[str]


class RecommendResponse(BaseModel):
    sort_by: str
    ranking: list[RecommendItem]
    excluded: list[ExcludedItem] = []
    warnings: list[str] = []


class MinMax(BaseModel):
    min: int
    max: int


class RangeResponse(BaseModel):
    item: str
    rain_mm: MinMax
    temp_c: MinMax
    pest_kg_ha: MinMax


class ItemListResponse(BaseModel):
    items: list[str]


# ---------- Logique ----------
def _resolve_item(bundle: dict, name: str) -> str | None:
    """Nom canonique de la culture (insensible à la casse), ou None si inconnue."""
    return next((i for i in bundle["items"] if i.lower() == name.strip().lower()), None)


def _bounds(bundle: dict, item: str) -> dict[str, tuple[int, int]]:
    """Bornes (min arrondi à l'entier inférieur, max à l'entier supérieur) d'une culture, ou 'all' = globales."""
    ranges = bundle["item_ranges"]
    sel = list(ranges.values()) if item == "all" else [ranges[item]]
    return {v: (math.floor(min(r[v]["min"] for r in sel)), math.ceil(max(r[v]["max"] for r in sel)))
            for v in VARS}


def _out_of_range(bundle: dict, item: str, ctx: Context, cols=VARS) -> list[str]:
    """Raisons pour lesquelles la saisie sort des bornes observées pour cette culture."""
    b = _bounds(bundle, item)
    return [f"{c}={getattr(ctx, c):g} hors plage observée [{b[c][0]}, {b[c][1]}]"
            for c in cols if not b[c][0] <= getattr(ctx, c) <= b[c][1]]


def _predict(bundle: dict, items: list[str], ctx: Context) -> pd.DataFrame:
    X = pd.DataFrame({"item": items, "rain_mm": ctx.rain_mm, "temp_c": ctx.temp_c,
                      "pest_kg_ha": ctx.pest_kg_ha})[bundle["features"]]
    out = pd.DataFrame({"item": items, "predicted_yield_t_ha": bundle["model"].predict(X)})
    out["norm_t_ha"] = out["item"].map(bundle["item_mean_t_ha"])
    out["gap_vs_norm_pct"] = (out["predicted_yield_t_ha"] / out["norm_t_ha"] - 1) * 100
    return out.round(3)


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/getitemlist", response_model=ItemListResponse)
def get_item_list(request: Request):
    """Toutes les cultures prises en compte."""
    return ItemListResponse(items=list(request.app.state.bundle["items"]))


@app.get("/getrange", response_model=RangeResponse)
def get_range(request: Request, item: str = Query(..., description="Culture, ou 'all' pour les bornes globales",
                                                  examples=["maize"])):
    """Min (entier inférieur) et max (entier supérieur) de pluie, température et pesticides observés."""
    b = request.app.state.bundle
    name = "all" if item.strip().lower() == "all" else _resolve_item(b, item)
    if name is None:
        raise HTTPException(404, f"Culture inconnue '{item}'. Valeurs possibles : {list(b['items'])} ou 'all'")
    return RangeResponse(item=name, **{v: MinMax(min=lo, max=hi) for v, (lo, hi) in _bounds(b, name).items()})


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    b = request.app.state.bundle
    item = _resolve_item(b, req.item)
    if item is None:
        raise HTTPException(422, f"Culture inconnue '{req.item}'. Valeurs possibles : {list(b['items'])}")
    row = _predict(b, [item], req).iloc[0]
    return PredictResponse(item=item, predicted_yield_t_ha=row.predicted_yield_t_ha,
                           norm_t_ha=row.norm_t_ha, gap_vs_norm_pct=row.gap_vs_norm_pct,
                           warnings=_out_of_range(b, item, req))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    b = request.app.state.bundle
    # on écarte les cultures dont la plage observée (pluie / température) ne couvre pas la saisie
    kept, excluded = [], []
    for it in b["items"]:
        why = _out_of_range(b, it, req, cols=("rain_mm", "temp_c"))
        (excluded.append(ExcludedItem(item=it, reasons=why)) if why else kept.append(it))

    ranking = []
    if kept:
        df = _predict(b, kept, req)
        key = "gap_vs_norm_pct" if req.sort_by == "ecart_vs_norme" else "predicted_yield_t_ha"
        df = df.sort_values(key, ascending=False).reset_index(drop=True)
        ranking = [RecommendItem(rank=i + 1, **r) for i, r in enumerate(df.to_dict("records"))]
    # pesticides : pas de filtre, simple avertissement si hors plage globale
    return RecommendResponse(sort_by=req.sort_by, ranking=ranking, excluded=excluded,
                             warnings=_out_of_range(b, "all", req, cols=("pest_kg_ha",)))
