# usage: pytest tests/test_unit.py
#        pytest

import pytest
from src.api import main as api

class FakeModel:
    def predict(self, X):
        return [10.0] * len(X)

@pytest.fixture
def bundle():
    return {
        "items": ["Maize", "Wheat"],
        "features": ["item", "rain_mm", "temp_c", "pest_kg_ha"],
        "item_mean_t_ha": {"Maize": 8.0, "Wheat": 5.0},
        "item_ranges": {
            "Maize": {"rain_mm": {"min": 100.4, "max": 2000.2},
                      "temp_c": {"min": 5.5, "max": 28.1},
                      "pest_kg_ha": {"min": 0.0, "max": 10.0}},
            "Wheat": {"rain_mm": {"min": 300.0, "max": 1500.0},
                      "temp_c": {"min": 2.0, "max": 20.0},
                      "pest_kg_ha": {"min": 0.5, "max": 8.0}},
        },
        "model": FakeModel(),
    }

def test_resolve_item_case_insensitive(bundle):
    assert api._resolve_item(bundle, "  maize ") == "Maize"

def test_resolve_item_unknown(bundle):
    assert api._resolve_item(bundle, "licorne") is None

def test_bounds_floor_and_ceil(bundle):
    b = api._bounds(bundle, "Maize")
    assert b["rain_mm"] == (100, 2001)
    assert b["temp_c"] == (5, 29)

def test_bounds_all_takes_global_extremes(bundle):
    b = api._bounds(bundle, "all")
    assert b["rain_mm"] == (100, 2001)
    assert b["temp_c"] == (2, 29)
    assert b["pest_kg_ha"] == (0, 10)

def test_out_of_range_detects_rain(bundle):
    ctx = api.Context(rain_mm=5000, temp_c=15, pest_kg_ha=2)
    reasons = api._out_of_range(bundle, "Maize", ctx)
    assert len(reasons) == 1 and "rain_mm" in reasons[0]

def test_in_range_no_warning(bundle):
    ctx = api.Context(rain_mm=1000, temp_c=15, pest_kg_ha=2)
    assert api._out_of_range(bundle, "Maize", ctx) == []

def test_predict_gap_vs_norm(bundle):
    ctx = api.Context(rain_mm=1000, temp_c=15, pest_kg_ha=2)
    df = api._predict(bundle, ["Maize", "Wheat"], ctx)
    assert df.loc[0, "gap_vs_norm_pct"] == 25.0    # 10 vs norme 8
    assert df.loc[1, "gap_vs_norm_pct"] == 100.0   # 10 vs norme 5
