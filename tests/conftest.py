import pytest
from fastapi.testclient import TestClient

from src.api.main import app

@pytest.fixture(scope="session")
def client():
    # "with" déclenche le lifespan : le vrai modèle est chargé une fois
    with TestClient(app) as c:
        yield c

@pytest.fixture
def ctx():
    return {"rain_mm": 1000, "temp_c": 15, "pest_kg_ha": 2}
