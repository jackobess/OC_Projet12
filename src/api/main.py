"""API de prédiction / recommandation de rendement agricole (Agritech Answers)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# /app/src/api/main.py -> ROOT = /app (idem en local : racine du projet)
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "yield_model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH)  # chargé une seule fois au démarrage
    yield


app = FastAPI(
    title="Agritech Answers - Yield API",
    description="Prédiction du rendement d'une culture et recommandation de la meilleure culture.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Schémas ----------
class Context(BaseModel):
    rain_mm: float = Field(..., ge=0, le=6000, description="Pluie annuelle (mm/an)", examples=[1000])
    temp_c: float = Field(..., ge=-10, le=40, description="Température moyenne (°C)", examples=[15])
    pest_kg_ha: float = Field(..., ge=0, le=100, description="Pesticides (kg matière active/ha)", examples=[2])


class PredictRequest(Context):
    item: str = Field(..., description="Culture (nom anglais, cf. GET /meta)", examples=["Maize"])


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


class RecommendResponse(BaseModel):
    sort_by: str
    ranking: list[RecommendItem]
    warnings: list[str] = []


# ---------- Logique ----------
def _predict(bundle: dict, items: list[str], ctx: Context) -> pd.DataFrame:
    X = pd.DataFrame({"item": items, "rain_mm": ctx.rain_mm, "temp_c": ctx.temp_c,
                      "pest_kg_ha": ctx.pest_kg_ha})[bundle["features"]]
    out = pd.DataFrame({"item": items, "predicted_yield_t_ha": bundle["model"].predict(X)})
    out["norm_t_ha"] = out["item"].map(bundle["item_mean_t_ha"])
    out["gap_vs_norm_pct"] = (out["predicted_yield_t_ha"] / out["norm_t_ha"] - 1) * 100
    return out.round(3)


def _range_warnings(bundle: dict, ctx: Context) -> list[str]:
    """Prévient si une saisie sort de la plage vue à l'entraînement (prédiction peu fiable)."""
    msgs = []
    for col, (lo, hi) in bundle["input_ranges"].items():
        v = getattr(ctx, col)
        if not lo <= v <= hi:
            msgs.append(f"{col}={v} hors plage d'entraînement [{lo:.1f}, {hi:.1f}] : résultat peu fiable")
    return msgs


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta(request: Request):
    """Cultures acceptées et plages d'entrée (utile pour alimenter les sliders du front)."""
    b = request.app.state.bundle
    return {"items": b["items"], "input_ranges": b["input_ranges"], "model": b["model_name"],
            "norm_t_ha": b["item_mean_t_ha"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    b = request.app.state.bundle
    if req.item not in b["items"]:
        raise HTTPException(422, f"Culture inconnue '{req.item}'. Valeurs possibles : {b['items']}")
    row = _predict(b, [req.item], req).iloc[0]
    return PredictResponse(item=req.item, predicted_yield_t_ha=row.predicted_yield_t_ha,
                           norm_t_ha=row.norm_t_ha, gap_vs_norm_pct=row.gap_vs_norm_pct,
                           warnings=_range_warnings(b, req))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request):
    b = request.app.state.bundle
    df = _predict(b, list(b["items"]), req)
    key = "gap_vs_norm_pct" if req.sort_by == "ecart_vs_norme" else "predicted_yield_t_ha"
    df = df.sort_values(key, ascending=False).reset_index(drop=True)
    ranking = [RecommendItem(rank=i + 1, **r) for i, r in enumerate(df.to_dict("records"))]
    return RecommendResponse(sort_by=req.sort_by, ranking=ranking, warnings=_range_warnings(b, req))
