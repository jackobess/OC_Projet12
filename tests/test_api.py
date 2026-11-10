# usage: pytest tests/test_api.py
#        pytest

import pytest
from fastapi.testclient import TestClient

from src.api import main as api

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_item_list(client):
    items = client.get("/getitemlist").json()["items"]
    assert "Maize" in items and len(items) >= 2

@pytest.mark.parametrize("item", ["all", "maize"])
def test_getrange(client, item):
    r = client.get("/getrange", params={"item": item})
    assert r.status_code == 200
    for v in ("rain_mm", "temp_c", "pest_kg_ha"):
        assert r.json()[v]["min"] <= r.json()[v]["max"]

def test_getrange_unknown_crop(client):
    assert client.get("/getrange", params={"item": "licorne"}).status_code == 404

def test_predict_ok(client, ctx):
    r = client.post("/predict", json={**ctx, "item": "Maize"})
    assert r.status_code == 200
    body = r.json()
    assert body["item"] == "Maize"
    assert body["predicted_yield_t_ha"] > 0

def test_predict_case_insensitive(client, ctx):
    r = client.post("/predict", json={**ctx, "item": "maize"})
    assert r.status_code == 200 and r.json()["item"] == "Maize"

def test_predict_unknown_crop(client, ctx):
    r = client.post("/predict", json={**ctx, "item": "Licorne"})
    assert r.status_code == 422

@pytest.mark.parametrize("bad", [
    {"rain_mm": -1}, {"temp_c": 99}, {"pest_kg_ha": 500},
])
def test_predict_invalid_values(client, ctx, bad):
    r = client.post("/predict", json={**ctx, **bad, "item": "Maize"})
    assert r.status_code == 422

def test_predict_missing_field(client, ctx):
    assert client.post("/predict", json=ctx).status_code == 422

def test_predict_warns_outside_observed_range(client, ctx):
    r = client.post("/predict", json={**ctx, "rain_mm": 6000, "item": "Maize"})
    assert r.status_code == 200
    assert any("rain_mm" in w for w in r.json()["warnings"])

def test_recommend_sorted_by_gap(client, ctx):
    ranking = client.post("/recommend", json=ctx).json()["ranking"]
    assert len(ranking) >= 1
    gaps = [x["gap_vs_norm_pct"] for x in ranking]
    assert gaps == sorted(gaps, reverse=True)
    assert [x["rank"] for x in ranking] == list(range(1, len(ranking) + 1))

def test_recommend_sorted_by_yield(client, ctx):
    r = client.post("/recommend", json={**ctx, "sort_by": "rendement"})
    ys = [x["predicted_yield_t_ha"] for x in r.json()["ranking"]]
    assert ys == sorted(ys, reverse=True)

def test_recommend_covers_every_crop_once(client, ctx):
    items = set(client.get("/getitemlist").json()["items"])
    body = client.post("/recommend", json=ctx).json()
    kept = {x["item"] for x in body["ranking"]}
    excluded = {x["item"] for x in body["excluded"]}
    assert kept | excluded == items and not kept & excluded

def test_recommend_extreme_context_excludes_all(client):
    r = client.post("/recommend", json={"rain_mm": 6000, "temp_c": 40, "pest_kg_ha": 2})
    body = r.json()
    assert r.status_code == 200 and body["ranking"] == []
    assert len(body["excluded"]) >= 1

def test_recommend_invalid_sort_by(client, ctx):
    r = client.post("/recommend", json={**ctx, "sort_by": "nimportequoi"})
    assert r.status_code == 422

# --- robustesse : modèle manquant => l'API refuse de démarrer, erreur claire ---
def test_startup_fails_if_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "absent.joblib")
    with pytest.raises(FileNotFoundError):
        with TestClient(api.app):
            pass
