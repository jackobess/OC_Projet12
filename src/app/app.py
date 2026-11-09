# Front-end Streamlit : aucune logique ML, uniquement des appels à l'API FastAPI.
#
# Usage: streamlit run src/app/app.py

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
import yaml
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# config.yaml à la racine du projet (src/app/app.py -> parents[2]) ; surchargeable via CONFIG_PATH
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", Path(__file__).resolve().parents[2] / "config.yaml"))
with open(CONFIG_PATH, encoding="utf-8") as f:
    FR = yaml.safe_load(f)["item_fr"]
EN = {v: k for k, v in FR.items()}

st.set_page_config(page_title="Agritech Answers - Rendements", page_icon="🌾", layout="wide")
st.markdown("""
<style>
div[data-baseweb="select"] > div { border-radius: 10px; border: 1px solid #2e7d32; }
div[data-baseweb="select"], div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="select"] input { caret-color: transparent; }  /* cache le curseur de saisie */

[data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    width: 100% !important;
    display: flex;
    justify-content: center;
}
[data-testid="stRadio"] {
    display: flex; flex-direction: row; align-items: center; gap: 1rem; width: fit-content;
}
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { margin-bottom: 0; }

[data-testid="stSliderTickBar"],
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] { opacity: 1 !important; visibility: visible !important; }

[data-testid="stElementContainer"]:has([data-testid="stSelectbox"]) {
    width: 100% !important;
    display: flex;
    justify-content: center;
}
[data-testid="stSelectbox"] {
    width: 320px;       /* ajuste */
    max-width: 100%;
}

div[data-baseweb="popover"] > div {
    border: 1px solid #2e7d32;
    border-radius: 10px;
    overflow: hidden;          /* pour que les coins arrondis coupent bien les options */
}

[data-testid="stElementContainer"]:has([data-testid="stButton"]) {
    width: 100% !important;
    display: flex;
    justify-content: center;
    margin-top: 1.5rem;       /* espace au-dessus, ajuste */
    margin-bottom: 1rem;
}
[data-testid="stButton"] {
    width: 320px;       /* ajuste */
    max-width: 100%;
}

.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 1rem; max-width: 1000px; margin: 0 auto;
}
h1, h2, h3 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------- Appels API ----------
@st.cache_data(ttl=300, show_spinner=False)
def get_items() -> list[str]:
    r = requests.get(f"{API_URL}/getitemlist", timeout=10)
    r.raise_for_status()
    return r.json()["items"]


@st.cache_data(ttl=300, show_spinner=False)
def get_range(item: str) -> dict:
    r = requests.get(f"{API_URL}/getrange", params={"item": item}, timeout=10)
    r.raise_for_status()
    return r.json()


def call_api(endpoint: str, payload: dict) -> dict:
    r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def context_inputs(rng: dict, scope: str) -> dict:
    """Sliders bornés par les plages observées ; les valeurs sont conservées d'un mode/culture à l'autre (clampées)."""
    store = st.session_state.setdefault("ctx", {})
    r, t, p = rng["rain_mm"], rng["temp_c"], rng["pest_kg_ha"]
    r_lo = int(math.floor(r["min"] / 10) * 10)   # inf à la dizaine
    r_hi = int(math.ceil(r["max"] / 10) * 10)    # sup à la dizaine
    def seed(name, key, default, lo, hi, cast):
        # init uniquement si le widget n'existe pas encore (nouveau scope) ; on reprend la dernière valeur clampée
        if key not in st.session_state:
            st.session_state[key] = cast(min(max(store.get(name, default), lo), hi))

    c1, c2, c3 = st.columns(3)
    with c1:
        seed("rain_mm", f"rain_{scope}", ((r_lo + r_hi) // 2 // 10) * 10, r_lo, r_hi, int)
        rain = st.slider("Pluie annuelle (mm)", r_lo, r_hi, step=10, key=f"rain_{scope}")
    with c2:
        seed("temp_c", f"temp_{scope}", round((t["min"] + t["max"]) / 2, 1),
             float(t["min"]), float(t["max"]), float)
        temp = st.slider("Température moyenne (°C)", float(t["min"]), float(t["max"]),
                         step=0.5, key=f"temp_{scope}")
    with c3:
        seed("pest_kg_ha", f"pest_{scope}", min(2.0, float(p["max"])),
             float(p["min"]), float(p["max"]), float)
        pest = st.slider("Pesticides (kg/ha)", float(p["min"]), float(p["max"]),
                         step=0.5, key=f"pest_{scope}")

    store.update(rain_mm=rain, temp_c=temp, pest_kg_ha=pest)
    return {"rain_mm": rain, "temp_c": temp, "pest_kg_ha": pest}

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(ttl=300, show_spinner="Calcul de la courbe...")
def yield_curve(item: str, rain: float, temp: float, p_max: float, step: float = 0.5) -> pd.DataFrame:
    """Rendement prédit pour chaque dose de pesticides (0 → p_max, pas de 0.5), reste du contexte fixe."""
    doses = np.arange(0.0, p_max + step / 2, step)

    def one(v):
        try:
            res = call_api("/predict", {"item": item, "rain_mm": rain, "temp_c": temp, "pest_kg_ha": float(v)})
            return {"Pesticides (kg/ha)": float(v), "Rendement prédit (t/ha)": res["predicted_yield_t_ha"]}
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:   # ~67 appels pour p_max=33, en parallèle c'est quasi instantané
        rows = [r for r in ex.map(one, doses) if r]
    return pd.DataFrame(rows)

# ---------- Page ----------
st.title("🌾 Aide à la décision agricole v1.13")                                                                         # Page Title

try:
    items = get_items()
except requests.RequestException:
    st.error(f"API injoignable ({API_URL}). Vérifie qu'elle tourne puis recharge la page.")
    st.stop()

mode = st.radio("Mode:", ["Prédiction", "Recommandation"], horizontal=True)
st.subheader("Quelle culture privilégier ?" if mode == "Recommandation" else "Quel rendement attendre ?")

st.divider()

# ---------- Mode Prédiction ----------
if mode == "Prédiction":
    culture = st.selectbox("Culturof", sorted(FR.get(i, i) for i in items),label_visibility="collapsed")
    item = EN.get(culture, culture)
    rng = get_range(item)
    ctx = context_inputs(rng, scope=item)
    p_max = float(rng["pest_kg_ha"]["max"])
    if ctx["pest_kg_ha"] >= p_max:
        st.html("""
            <style>
            [class*="st-key-pest_"] [data-testid="stSliderThumbValue"] { font-size: 0; white-space: nowrap; }
            [class*="st-key-pest_"] [data-testid="stSliderThumbValue"]::after {
                content: "📈 Rendement/Pesticides";
                font-size: 0.875rem;
                position: relative;
                top: -1.5rem;      /* monte le texte, ajuste (-0.4 à -1rem) */
           }
            </style>
        """)

    if st.button("Prédire", type="primary"):
        try:
            res = call_api("/predict", {"item": item, **ctx})
        except requests.RequestException as e:
            st.error(f"Erreur API : {e}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Rendement prédit", f"{res['predicted_yield_t_ha']:.2f} t/ha")
            c2.metric("Écart vs rendement moyen de la culture", f"{res['gap_vs_norm_pct']:+.1f} %",
                      help=f"Rendement moyen historique : {res['norm_t_ha']:.2f} t/ha")
            for w in res["warnings"]:
                st.warning(w)

            # ---- courbe rendement vs pesticides ----
            p_max = float(rng["pest_kg_ha"]["max"])
            if ctx["pest_kg_ha"] >= p_max - 0.25:       # ta condition : pest au max
                curve = yield_curve(item, ctx["rain_mm"], ctx["temp_c"], p_max)
                if not curve.empty:
                    x, y = "Pesticides (kg/ha)", "Rendement prédit (t/ha)"
                    line = alt.Chart(curve).mark_line().encode(
                        x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
                        tooltip=[x, y])
                    st.subheader("Rendement selon la dose de pesticides")
                    st.altair_chart(line.properties(height=300), width="stretch")
                    st.caption("Courbe indicative: Le modèle a été entraîné sur des doses de pesticides annuelles par pays, "
                               "toutes cultures confondues. La courbe ne reflète donc pas un effet propre à la culture choisie.")

# ---------- Mode Recommandation ----------
else:
    ctx = context_inputs(get_range("all"), scope="all")
    tri = st.radio("Classer par", ["Écart vs rendement moyen (%)", "Rendement (t/ha)"], horizontal=True)
    sort_by = "ecart_vs_norme" if tri.startswith("Écart") else "rendement"

    if st.button("Recommander", type="primary"):
        try:
            res = call_api("/recommend", {**ctx, "sort_by": sort_by})
        except requests.RequestException as e:
            st.error(f"Erreur API : {e}")
        else:
            if not res["ranking"]:
                st.warning("Aucune culture n'est adaptée à ces conditions (pluie / température hors des plages observées).")
            else:
                df = pd.DataFrame(res["ranking"])
                df["Culture"] = df["item"].map(lambda i: FR.get(i, i))
                df = df.rename(columns={"rank": "Rang", "predicted_yield_t_ha": "Rendement prédit (t/ha)",
                                        "norm_t_ha": "Rendement moyen (t/ha)", "gap_vs_norm_pct": "Écart vs moyenne (%)"})
                y_col = "Écart vs moyenne (%)" if sort_by == "ecart_vs_norme" else "Rendement prédit (t/ha)"

                st.success(f"Culture recommandée : **{df.iloc[0]['Culture']}**")
                chart = (alt.Chart(df).mark_bar()
                         .encode(x=alt.X(f"{y_col}:Q"), y=alt.Y("Culture:N", sort=None, title=None),
                                 tooltip=["Culture", "Rendement prédit (t/ha)", "Écart vs moyenne (%)"])
                         .properties(height=380))
                st.altair_chart(chart, width="stretch")
                st.dataframe(df[["Rang", "Culture", "Rendement prédit (t/ha)", "Rendement moyen (t/ha)",
                                 "Écart vs moyenne (%)"]], hide_index=True, width="stretch")
            if res["excluded"]:
                with st.expander(f"{len(res['excluded'])} culture(s) écartée(s) : conditions hors plage observée"):
                    for e in res["excluded"]:
                        st.markdown(f"- **{FR.get(e['item'], e['item'])}** : " + " ; ".join(e["reasons"]))
            for w in res["warnings"]:
                st.warning(w)
