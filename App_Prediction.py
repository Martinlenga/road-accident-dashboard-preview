"""
ROAD SAFETY INTELLIGENCE
UK STATS19 collision analytics and research-only severity prediction.

Run:
    streamlit run App_Prediction.py

Expected project layout:
    App_Prediction.py
    data/
        *collision*.csv
        *casualt*.csv
        *vehicle*.csv
    models/
        stats19_severity_model.joblib
        model_contract.json
        metadata.json

The model predicts severity conditional on a recorded collision. It does not
predict whether a collision will occur and is not approved for operational use.
"""

from __future__ import annotations

import json
import inspect
import warnings
import sys
import importlib.util
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# =============================================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# =============================================================================

st.set_page_config(
    page_title="Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

COLLISION_FILE_KEY = "collision"
CASUALTY_FILE_KEY = "casualt"
VEHICLE_FILE_KEY = "vehicle"

SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight",
                   "1": "Fatal", "2": "Serious", "3": "Slight"}
SEVERITY_COLOURS = {"Fatal": "#FF5C70", "Serious": "#FFB547", "Slight": "#35C6A5"}

AREA_LABELS = {1: "Urban", 2: "Rural", 3: "Unallocated",
               "1": "Urban", "2": "Rural", "3": "Unallocated"}
ROAD_TYPE_LABELS = {
    1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
    6: "Single carriageway", 7: "Slip road", 9: "Unknown",
    12: "One-way street / roundabout",
}
LIGHT_LABELS = {
    1: "Daylight", 4: "Darkness – lights lit",
    5: "Darkness – lights unlit", 6: "Darkness – no lighting",
    7: "Darkness – lighting unknown", -1: "Unknown",
}
WEATHER_LABELS = {
    1: "Fine – no high winds", 2: "Raining – no high winds",
    3: "Snowing – no high winds", 4: "Fine – high winds",
    5: "Raining – high winds", 6: "Snowing – high winds",
    7: "Fog or mist", 8: "Other weather", 9: "Unknown weather",
    -1: "Unknown",
}
SURFACE_LABELS = {
    1: "Dry", 2: "Wet or damp", 3: "Snow",
    4: "Frost or ice", 5: "Flood over 3 cm deep",
    6: "Oil or diesel", 7: "Mud", 9: "Unknown",
    -1: "Unknown",
}
JUNCTION_DETAIL_LABELS = {
    0: "Not at/near junction", 1: "Roundabout", 2: "Mini-roundabout",
    3: "T or staggered junction", 5: "Slip road", 6: "Crossroads",
    7: "More than four arms", 8: "Private drive/entrance",
    9: "Other junction", 99: "Unknown",
}
JUNCTION_CONTROL_LABELS = {
    0: "No control/not at junction", 1: "Authorised person",
    2: "Traffic signal", 3: "Stop sign", 4: "Give way/uncontrolled",
}
ROAD_CLASS_LABELS = {
    1: "Motorway", 2: "A(M)", 3: "A road",
    4: "B road", 5: "C road", 6: "Unclassified",
}
SEX_LABELS = {1: "Male", 2: "Female", 3: "Not applicable",
              9: "Unknown", -1: "Unknown"}
CASUALTY_CLASS_LABELS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
AGE_BAND_LABELS = {
    1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20", 5: "21–25",
    6: "26–35", 7: "36–45", 8: "46–55", 9: "56–65",
    10: "66–75", 11: "Over 75", -1: "Unknown",
}
AGE_BAND_ORDER = ["0–5", "6–10", "11–15", "16–20", "21–25",
                  "26–35", "36–45", "46–55", "56–65", "66–75",
                  "Over 75", "Unknown"]
VEHICLE_TYPE_LABELS = {
    1: "Pedal cycle", 2: "Motorcycle ≤50cc", 3: "Motorcycle ≤125cc",
    4: "Motorcycle 125–500cc", 5: "Motorcycle >500cc",
    8: "Taxi/private hire", 9: "Car", 10: "Minibus",
    11: "Bus or coach", 16: "Ridden horse", 17: "Agricultural vehicle",
    18: "Tram", 19: "Van/goods ≤3.5t", 20: "Goods 3.5–7.5t",
    21: "Goods >7.5t", 22: "Mobility scooter", 23: "Electric motorcycle",
    90: "Other vehicle", 97: "Motorcycle – cc unknown",
    98: "Goods – weight unknown", 99: "Unknown vehicle", -1: "Unknown",
}
USER_GROUP = {
    0: "Pedestrian", 1: "Cyclist", 8: "Car occupant", 9: "Car occupant",
    10: "Bus/coach occupant", 11: "Bus/coach occupant",
    19: "Van/goods occupant", 20: "Van/goods occupant",
    21: "Van/goods occupant", 98: "Van/goods occupant",
}
USER_GROUP.update({x: "Motorcyclist" for x in (2, 3, 4, 5, 23, 97, 103, 104, 105, 106)})
VULNERABLE_USERS = ["Pedestrian", "Cyclist", "Motorcyclist"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_ORDER = [f"{h:02d}:00" for h in range(24)]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

INK = "#EAF1FA"
MUTED = "#9AAAC0"
BG = "#10151D"
PANEL = "#1A222D"
PANEL_ALT = "#222D3A"
BORDER = "#354252"
BLUE = "#66A6FF"
TEAL = "#35C6A5"
AMBER = "#F4B860"
RED = "#F06C79"
PURPLE = "#B39DDB"
GREEN = "#77C98B"
CORAL = "#F28C68"
CYAN = "#63C7D8"
YEAR_PALETTE = [BLUE, TEAL, AMBER, PURPLE, RED]

RESEARCH_ONLY_NOTICE = (
    "RESEARCH / DEMONSTRATION ONLY — this model has not passed the project's "
    "acceptance gate. Do not use its predictions for operational road-safety, "
    "emergency response, enforcement, insurance or individual decisions."
)


# =============================================================================
# 2. MODEL SERIALIZATION COMPATIBILITY
# Keep these class names and constructor signatures stable: joblib may need
# them when loading the saved estimator.
# =============================================================================

class CategoricalStringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.categorical_columns_ = [
            c for c in self.categorical_columns if c in X.columns
        ]
        return self

    def transform(self, X):
        transformed = X.copy()
        columns = getattr(self, "categorical_columns_", self.categorical_columns)
        for column in columns:
            if column in transformed.columns:
                values = transformed[column].astype(object)
                transformed[column] = values.where(values.notna(), "Missing").map(str)
        return transformed


class CatBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    def __init__(
        self, categorical_columns, iterations=350, depth=6,
        learning_rate=0.08, loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro", auto_class_weights="Balanced",
        random_seed=42, thread_count=-1, verbose=False,
        allow_writing_files=False,
    ):
        self.categorical_columns = categorical_columns
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files

    def fit(self, X, y, sample_weight=None):
        from catboost import CatBoostClassifier
        cat_features = [c for c in self.categorical_columns if c in X.columns]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, loss_function=self.loss_function,
            eval_metric=self.eval_metric, cat_features=cat_features,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_seed, thread_count=self.thread_count,
            verbose=self.verbose, allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class DecisionAdjustedModel:
    def __init__(self, fitted_estimator, class_multipliers):
        self.fitted_estimator = fitted_estimator
        self.class_multipliers = dict(class_multipliers)
        self.classes_ = np.asarray(fitted_estimator.classes_)

    def predict_proba(self, X):
        return self.fitted_estimator.predict_proba(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        multipliers = np.asarray(
            [self.class_multipliers.get(label, 1.0) for label in self.classes_],
            dtype=float,
        )
        return self.classes_[np.argmax(probabilities * multipliers, axis=1)]


# =============================================================================
# 3. GLOBAL VISUAL DESIGN
# =============================================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
      color-scheme: dark;
      --bg: {BG}; --panel: {PANEL}; --panel2: {PANEL_ALT};
      --ink: {INK}; --muted: {MUTED}; --border: {BORDER};
      --blue: {BLUE}; --teal: {TEAL}; --amber: {AMBER}; --red: {RED};
    }}
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
      font-family: 'DM Sans', sans-serif !important;
      background: {BG} !important;
      color: {INK} !important;
    }}
    .stApp {{ background: {BG} !important; }}
    [data-testid="stHeader"] {{ background: rgba(8,19,33,.92) !important; }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    .block-container {{ max-width: 1600px; padding-top: 1.35rem; padding-bottom: 3rem; }}
    h1,h2,h3,h4,h5,p,li,label {{ color: {INK}; }}
    h1,h2,h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }}
    h1 {{ font-size: clamp(2rem, 3vw, 2.8rem) !important; font-weight: 700 !important; }}
    h2 {{ font-size: 1.55rem !important; }}
    h3 {{ font-size: 1.12rem !important; }}
    p, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg,#151F2B 0%,#111923 100%) !important; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .72rem; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding: 8px 10px; border-radius: 9px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: rgba(99,199,216,.10); }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{ border:1px solid {BORDER} !important; border-radius:10px !important; }}
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{ background:{PANEL_ALT} !important; }}
    [data-baseweb="popover"] [role="option"] {{ color:{INK} !important; }}
    [data-testid="stSidebar"] * {{ color: {INK} !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{ background: {PANEL_ALT} !important; color: {INK} !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: {MUTED} !important; }}
    [data-testid="stMetric"] {{ min-height:118px; display:flex; flex-direction:column; justify-content:center;
      
      background: linear-gradient(145deg, {PANEL_ALT}, {PANEL}) !important;
      border: 1px solid {BORDER} !important; border-radius: 14px !important;
      padding: 16px 18px !important;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .78rem !important; }}
    [data-testid="stMetricValue"] {{ color: {INK} !important; font-family: 'Space Grotesk',sans-serif; }}
    [data-testid="stMetricDelta"] {{ color: {TEAL} !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background: {PANEL} !important; border: 1px solid {BORDER} !important;
      border-radius: 16px !important;
    }}
    [data-testid="stExpander"] {{ background: {PANEL} !important; border: 1px solid {BORDER} !important; border-radius: 12px !important; }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
      background: {PANEL} !important; border: 1px solid {BORDER}; border-radius: 12px;
    }}
    [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}
    button[data-baseweb="tab"] {{ color: {MUTED} !important; background: {PANEL} !important; border-radius: 10px 10px 0 0; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEAL} !important; border-bottom: 2px solid {TEAL} !important; }}
    .hero {{
      background: radial-gradient(circle at 85% 10%, rgba(53,198,165,.17), transparent 32%),
                  linear-gradient(120deg, #10233A 0%, #0C1A2C 62%, #10263A 100%);
      border: 1px solid {BORDER}; border-radius: 20px; padding: 26px 30px;
      margin-bottom: 22px; position: relative; overflow: hidden;
    }}
    .eyebrow {{ color: {TEAL}; text-transform: uppercase; letter-spacing: .17em; font-size: .7rem; font-weight: 800; }}
    .hero-title {{ color: {INK}; font-family: 'Space Grotesk',sans-serif; font-size: clamp(1.8rem,3vw,2.55rem); font-weight: 700; line-height: 1.12; margin: 8px 0; }}
    .hero-sub {{ color: #B3C2D5; font-size: .94rem; max-width: 780px; line-height: 1.65; }}
    .section-kicker {{ color: {TEAL}; font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 4px; }}
    .section-title {{ color: {INK}; font-family: 'Space Grotesk',sans-serif; font-size: 1.55rem; font-weight: 700; margin: 0 0 4px; }}
    .section-sub {{ color: {MUTED}; font-size: .88rem; margin-bottom: 18px; }}
    .panel-title {{ color: {INK}; font-family: 'Space Grotesk',sans-serif; font-size: 1rem; font-weight: 650; margin-bottom: 3px; }}
    .panel-note {{ color: {MUTED}; font-size: .76rem; margin-bottom: 12px; }}
    .notice {{
      background: rgba(255,181,71,.09); border: 1px solid rgba(255,181,71,.38);
      border-left: 4px solid {AMBER}; border-radius: 12px; color: #FFE0A6;
      padding: 13px 16px; font-size: .82rem; line-height: 1.55; margin: 12px 0 18px;
    }}
    .result-card {{ background: linear-gradient(135deg,#263A3A,#202B37); border:1px solid {BORDER}; border-radius:16px; padding:20px 22px; }}
    .result-label {{ color:{MUTED}; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
    .result-value {{ font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:700; margin-top:5px; }}
    .small-muted {{ color:{MUTED}; font-size:.76rem; }}
    .recommendation-card {{ height:100%; min-height:176px; margin-bottom:18px; background:linear-gradient(145deg,#222D3A,#1A222D); border:1px solid {BORDER}; border-radius:14px; padding:17px 18px; }}
    .recommendation-tag {{ font-size:.67rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:8px; }}
    .recommendation-title {{ color:{INK}; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:.98rem; margin-bottom:7px; }}
    .recommendation-copy {{ color:#B8C4D2; font-size:.81rem; line-height:1.55; }}
    .filter-caption {{ color:{MUTED}; font-size:.72rem; line-height:1.45; }}
    .footer {{ border-top:1px solid {BORDER}; margin-top:38px; padding-top:16px; color:{MUTED}; font-size:.74rem; }}
    div.stButton > button[kind="primary"], button[kind="primary"] {{
      background: {TEAL} !important; color: #062018 !important; border: 0 !important; font-weight: 800 !important;
      border-radius: 10px !important;
    }}
    div.stButton > button, .stDownloadButton > button {{
      border: 1px solid {BORDER} !important; border-radius: 10px !important;
      color: {INK} !important; background: {PANEL_ALT} !important;
    }}
    [data-testid="stForm"] {{ border:1px solid {BORDER}; border-radius:14px; padding:18px; background:{PANEL}; }}
    input, textarea {{ color: {INK} !important; }}
    /* Streamlit columns naturally wrap on narrow screens; these rules keep
       charts, cards and sidebar controls usable at tablet/mobile widths. */
    [data-testid="stHorizontalBlock"] {{ align-items: stretch; gap: 1.15rem; }}
    [data-testid="stColumn"] {{ min-width: 0 !important; }}
    [data-testid="stVegaLiteChart"] {{ width:100%; overflow:visible; }}
    .recommendation-card {{ display:flex; flex-direction:column; }}
    @media(max-width: 900px) {{
      .block-container {{ padding: 1rem 1rem 2rem; }}
      .hero {{ padding:22px; }}
      [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
      [data-testid="stHorizontalBlock"] > [data-testid="column"] {{ min-width:min(100%, 320px) !important; flex:1 1 100%; }}
    }}
    @media(max-width: 600px) {{
      .block-container {{ padding: .7rem .65rem 1.5rem; }}
      .hero {{ padding:17px; border-radius:15px; }}
      .hero-title {{ font-size:1.55rem; }}
      .section-title {{ font-size:1.3rem; }}
      [data-testid="stMetric"] {{ min-height:96px; padding:12px !important; }}
      [data-testid="stMetricValue"] {{ font-size:1.35rem !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 4. DATA LOADING AND PREPARATION
# =============================================================================

COLLISION_COLUMNS = [
    "collision_index", "date", "time", "day_of_week", "collision_severity",
    "urban_or_rural_area", "speed_limit", "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions", "junction_detail",
    "junction_control", "first_road_class", "second_road_class",
    "special_conditions_at_site", "carriageway_hazards", "latitude",
    "longitude", "number_of_vehicles", "number_of_casualties",
]
CASUALTY_COLUMNS = [
    "collision_index", "casualty_severity", "casualty_class",
    "age_band_of_casualty", "sex_of_casualty", "casualty_type",
]
VEHICLE_COLUMNS = [
    "collision_index", "sex_of_driver", "age_band_of_driver", "vehicle_type",
    "vehicle_manoeuvre", "engine_capacity_cc",
]


def find_csvs(keyword: str) -> list[Path]:
    """Find all matching annual CSVs, preferring the data/ directory."""
    for folder in [DATA_DIR, APP_DIR]:
        if folder.exists():
            matches = sorted(
                p for p in folder.glob("*.csv")
                if keyword in p.name.lower()
            )
            if matches:
                return matches
    raise FileNotFoundError(
        f"Could not find a CSV containing '{keyword}' in "
        f"{DATA_DIR} or {APP_DIR}."
    )


def read_selected_csv(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    """Read and combine matching annual exports using only required columns."""
    frames = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        selected = [column for column in columns if column in header]
        if "collision_index" not in selected:
            raise ValueError(f"{path.name} does not contain collision_index.")
        frames.append(pd.read_csv(path, usecols=selected, low_memory=False))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Annual exports can overlap at release boundaries; remove exact duplicate
    # rows without collapsing multiple casualty/vehicle records per collision.
    return combined.drop_duplicates().reset_index(drop=True)


def _map_code(series: pd.Series, mapping: dict, fallback="Unknown") -> pd.Series:
    numeric_map = {int(k): v for k, v in mapping.items() if str(k).lstrip("-").isdigit()}
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(numeric_map)
    return mapped.fillna(fallback)


@st.cache_data(show_spinner="Loading and preparing the Stats19 datasets...")
def load_stats19_data():
    collision_paths = find_csvs(COLLISION_FILE_KEY)
    casualty_paths = find_csvs(CASUALTY_FILE_KEY)
    vehicle_paths = find_csvs(VEHICLE_FILE_KEY)

    collisions = read_selected_csv(collision_paths, COLLISION_COLUMNS)
    casualties = read_selected_csv(casualty_paths, CASUALTY_COLUMNS)
    vehicles = read_selected_csv(vehicle_paths, VEHICLE_COLUMNS)

    for frame in (collisions, casualties, vehicles):
        frame["collision_index"] = frame["collision_index"].astype("string").str.strip()

    collisions["date"] = pd.to_datetime(
        collisions["date"], dayfirst=True, errors="coerce"
    )
    collisions["time"] = collisions["time"].astype("string").fillna("Unknown").str.strip()
    collisions["collision_severity_label"] = _map_code(
        collisions["collision_severity"], SEVERITY_LABELS
    )
    collisions["area_label"] = _map_code(collisions["urban_or_rural_area"], AREA_LABELS)
    collisions["road_type_label"] = _map_code(collisions["road_type"], ROAD_TYPE_LABELS)
    collisions["light_label"] = _map_code(collisions["light_conditions"], LIGHT_LABELS)
    collisions["weather_label"] = _map_code(collisions["weather_conditions"], WEATHER_LABELS)
    collisions["surface_label"] = _map_code(
        collisions["road_surface_conditions"], SURFACE_LABELS
    )
    collisions["junction_detail_label"] = _map_code(
        collisions["junction_detail"], JUNCTION_DETAIL_LABELS
    )
    collisions["junction_control_label"] = _map_code(
        collisions["junction_control"], JUNCTION_CONTROL_LABELS
    )
    collisions["first_road_class_label"] = _map_code(
        collisions["first_road_class"], ROAD_CLASS_LABELS
    )
    collisions["second_road_class_label"] = _map_code(
        collisions["second_road_class"], ROAD_CLASS_LABELS
    )

    speed = pd.to_numeric(collisions.get("speed_limit"), errors="coerce")
    collisions["speed_label"] = np.where(
        speed.notna() & speed.gt(0), speed.fillna(0).astype(int).astype(str), "Unknown"
    )
    collisions["weekday"] = collisions["date"].dt.day_name().fillna("Unknown")
    collisions["month_number"] = collisions["date"].dt.month
    collisions["month_label"] = collisions["date"].dt.strftime("%b").fillna("Unknown")
    collisions["year_month"] = collisions["date"].dt.to_period("M").astype("string")
    hour = pd.to_numeric(collisions["time"].str[:2], errors="coerce")
    collisions["hour"] = hour.where(hour.between(0, 23))
    collisions["hour_label"] = collisions["hour"].map(
        lambda x: f"{int(x):02d}:00" if pd.notna(x) else "Unknown"
    )
    collisions["is_fatal"] = pd.to_numeric(
        collisions["collision_severity"], errors="coerce"
    ).eq(1)
    collisions["is_dark"] = pd.to_numeric(
        collisions["light_conditions"], errors="coerce"
    ).isin([4, 5, 6, 7])

    casualties["casualty_severity_label"] = _map_code(
        casualties["casualty_severity"], SEVERITY_LABELS
    )
    casualties["casualty_class_label"] = _map_code(
        casualties["casualty_class"], CASUALTY_CLASS_LABELS
    )
    casualties["age_band_label"] = _map_code(
        casualties["age_band_of_casualty"], AGE_BAND_LABELS
    )
    casualties["sex_label"] = _map_code(casualties["sex_of_casualty"], SEX_LABELS)
    casualties["casualty_type_label"] = _map_code(
        casualties["casualty_type"], VEHICLE_TYPE_LABELS
    ) if "casualty_type" in casualties.columns else "Unknown"

    if "casualty_type" in casualties.columns:
        casualty_type = pd.to_numeric(casualties["casualty_type"], errors="coerce")
        casualties["user_type"] = casualty_type.map(USER_GROUP).fillna("Other")
        casualties.loc[casualty_type.eq(0), "user_type"] = "Pedestrian"
    else:
        casualties["user_type"] = np.where(
            pd.to_numeric(casualties["casualty_class"], errors="coerce").eq(3),
            "Pedestrian", "Other"
        )
    casualties["is_ksi"] = pd.to_numeric(
        casualties["casualty_severity"], errors="coerce"
    ).isin([1, 2])
    casualties["is_vru"] = casualties["user_type"].isin(VULNERABLE_USERS)

    vehicles["driver_sex_label"] = _map_code(vehicles["sex_of_driver"], SEX_LABELS)
    vehicles["driver_age_band_label"] = _map_code(
        vehicles["age_band_of_driver"], AGE_BAND_LABELS
    )
    vehicles["vehicle_type_label"] = _map_code(
        vehicles["vehicle_type"], VEHICLE_TYPE_LABELS
    )
    manoeuvre_labels = {
        1: "Reversing", 2: "Parked", 3: "Waiting/held up",
        4: "Slowing/stopping", 5: "Moving off", 6: "U-turn",
        7: "Turning left", 8: "Waiting to turn left", 9: "Turning right",
        10: "Waiting to turn right", 11: "Changing lane left",
        12: "Changing lane right", 13: "Overtaking moving vehicle",
        14: "Overtaking stationary vehicle", 15: "Overtaking nearside",
        16: "Going ahead – left bend", 17: "Going ahead – right bend",
        18: "Going ahead – other", 99: "Unknown",
    }
    vehicles["vehicle_manoeuvre_label"] = _map_code(
        vehicles["vehicle_manoeuvre"], manoeuvre_labels
    ) if "vehicle_manoeuvre" in vehicles.columns else "Unknown"

    return collisions, casualties, vehicles


def filter_linked_data(collisions, casualties, vehicles):
    ids = set(collisions["collision_index"].dropna().astype(str))
    casualty_view = casualties[
        casualties["collision_index"].astype(str).isin(ids)
    ].copy()
    vehicle_view = vehicles[
        vehicles["collision_index"].astype(str).isin(ids)
    ].copy()
    return casualty_view, vehicle_view


# =============================================================================
# 5. REUSABLE CHARTS, METRICS AND INSIGHTS
# =============================================================================

def panel(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-note">{subtitle}</div>', unsafe_allow_html=True)


def section_heading(kicker: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def show_chart(chart, key: str | None = None, height: int | None = None):
    if height:
        chart = chart.properties(height=height)
    chart = (
        chart.configure(background=PANEL)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER,
            domainColor=BORDER, tickColor=BORDER, labelFont="DM Sans",
            titleFont="DM Sans",
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, labelFont="DM Sans")
        .configure_title(color=INK, font="Space Grotesk")
    )
    kwargs = {"use_container_width": True}
    if key:
        kwargs["key"] = key
    st.altair_chart(chart, **kwargs)


def empty_state(message: str):
    st.info(message)


def count_frame(series: pd.Series, label="Category", value="Count"):
    result = series.astype("string").fillna("Unknown").value_counts().rename_axis(label).reset_index(name=value)
    return result


def severity_donut(df: pd.DataFrame, column: str, title: str = "Collision severity"):
    panel(title, "Share of collisions in the current selection.")
    if df.empty:
        return empty_state("No records are available for this chart.")
    counts = count_frame(df[column].astype(str), "Severity", "Collisions")
    counts = counts[counts["Severity"].isin(SEVERITY_ORDER)]
    chart = alt.Chart(counts).mark_arc(innerRadius=76, outerRadius=108, stroke=PANEL).encode(
        theta=alt.Theta("Collisions:Q"),
        color=alt.Color(
            "Severity:N",
            scale=alt.Scale(domain=SEVERITY_ORDER,
                            range=[SEVERITY_COLOURS[x] for x in SEVERITY_ORDER]),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=["Severity:N", alt.Tooltip("Collisions:Q", format=",")],
    ).properties(height=340, padding={"left": 12, "right": 12, "top": 12, "bottom": 18})
    show_chart(chart)


def bar_chart(df, category, value, title, colour=BLUE, horizontal=False, order=None, value_title="Records"):
    panel(title)
    if df.empty:
        return empty_state("No records are available for this chart.")
    if horizontal:
        enc = {
            "x": alt.X(f"{value}:Q", title=value_title),
            "y": alt.Y(f"{category}:N", sort=order or "-x", title=None),
        }
    else:
        enc = {
            "x": alt.X(f"{category}:N", sort=order or "-y", title=None,
                       axis=alt.Axis(labelAngle=-35 if df[category].nunique() > 7 else 0)),
            "y": alt.Y(f"{value}:Q", title=value_title),
        }
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=5, color=colour).encode(
        **enc,
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Category"),
            alt.Tooltip(f"{value}:Q", title=value_title, format=","),
        ],
    ).properties(height=340)
    show_chart(chart)


def stacked_severity_chart(df, category, count_col="Count", title="Severity by category"):
    panel(title, "Within-category severity composition.")
    if df.empty:
        return empty_state("No records are available for this chart.")

    # Source frames use either collision_severity_label or
    # casualty_severity_label. Normalise both to the chart's Severity field.
    severity_field = next(
        (c for c in ["Severity", "collision_severity_label",
                     "casualty_severity_label", "severity_label"] if c in df.columns),
        None,
    )
    if severity_field is None or category not in df.columns or count_col not in df.columns:
        return empty_state("The required category or severity fields are not available in this dataset.")
    plot_df = df.rename(columns={severity_field: "Severity"}).copy()
    plot_df = plot_df[plot_df["Severity"].isin(SEVERITY_ORDER)]
    if plot_df.empty:
        return empty_state("No recognised Fatal, Serious or Slight records are available for this selection.")

    # Stable category ordering improves readability for age bands and other
    # ordered dimensions; otherwise categories are sorted by total volume.
    if category == "age_band_label":
        category_sort = AGE_BAND_ORDER
    else:
        category_sort = (plot_df.groupby(category, observed=True)[count_col]
                         .sum().sort_values(ascending=False).index.tolist())
    chart = alt.Chart(plot_df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{count_col}:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share"),
        y=alt.Y(f"{category}:N", sort=category_sort, title=None),
        color=alt.Color(
            "Severity:N", sort=SEVERITY_ORDER,
            scale=alt.Scale(domain=SEVERITY_ORDER,
                            range=[SEVERITY_COLOURS[x] for x in SEVERITY_ORDER]),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Category"),
            "Severity:N",
            alt.Tooltip(f"{count_col}:Q", title="Records", format=","),
        ],
    ).properties(height=max(360, min(520, 34 * len(category_sort) + 90)),
                 padding={"left": 8, "right": 18, "top": 8, "bottom": 18})
    show_chart(chart)


def make_kpis(collisions, casualties):
    n_collisions = len(collisions)
    n_casualties = len(casualties)
    fatal = int(pd.to_numeric(collisions["collision_severity"], errors="coerce").eq(1).sum())
    killed = int(pd.to_numeric(casualties["casualty_severity"], errors="coerce").eq(1).sum())
    serious = int(pd.to_numeric(casualties["casualty_severity"], errors="coerce").eq(2).sum())
    vru_share = float(casualties["is_vru"].mean() * 100) if n_casualties else 0
    dark_share = float(collisions["is_dark"].mean() * 100) if n_collisions else 0
    return n_collisions, n_casualties, fatal, killed, serious, vru_share, dark_share


def render_kpis(collisions, casualties):
    values = make_kpis(collisions, casualties)
    labels = [
        ("COLLISIONS", f"{values[0]:,}", BLUE),
        ("CASUALTIES", f"{values[1]:,}", PURPLE),
        ("FATAL COLLISIONS", f"{values[2]:,}", RED),
        ("PEOPLE KILLED", f"{values[3]:,}", CORAL),
        ("SERIOUSLY INJURED", f"{values[4]:,}", AMBER),
        ("VULNERABLE ROAD USERS", f"{values[5]:.1f}%", CYAN),
    ]
    cols = st.columns(6)
    for col, (label, value, colour) in zip(cols, labels):
        with col:
            st.markdown(
                f'<div style="background:{PANEL};border:1px solid {BORDER};'
                f'border-top:3px solid {colour};border-radius:13px;padding:16px 15px;min-height:116px;height:116px;display:flex;flex-direction:column;justify-content:center">'
                f'<div style="color:{MUTED};font-size:.68rem;font-weight:800;letter-spacing:.08em">{label}</div>'
                f'<div style="color:{INK};font-family:Space Grotesk;font-size:1.65rem;font-weight:700;margin-top:7px">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def monthly_counts(df):
    """Return monthly counts by calendar year for an honest year comparison."""
    valid = df.dropna(subset=["date"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["MonthNo", "Month", "Year", "Collisions"])
    valid["Year"] = valid["date"].dt.year.astype(str)
    valid["MonthNo"] = valid["date"].dt.month
    valid["Month"] = valid["date"].dt.strftime("%b")
    return (valid.groupby(["Year", "MonthNo", "Month"], observed=True)
            .size().reset_index(name="Collisions")
            .sort_values(["Year", "MonthNo"]))


def generate_key_findings(collisions, casualties):
    findings = []
    if not collisions.empty:
        severity_share = collisions["collision_severity_label"].value_counts(normalize=True) * 100
        if "Fatal" in severity_share:
            findings.append(
                f"Fatal collisions account for {severity_share['Fatal']:.2f}% of the selected collision records."
            )
        valid_speed = collisions[collisions["speed_label"] != "Unknown"]
        if not valid_speed.empty:
            by_speed = valid_speed.groupby("speed_label").agg(
                collisions=("is_fatal", "size"), fatal=("is_fatal", "sum")
            )
            by_speed = by_speed[by_speed["collisions"] >= 100]
            if not by_speed.empty:
                by_speed["fatal_rate"] = by_speed["fatal"] / by_speed["collisions"] * 1000
                top = by_speed["fatal_rate"].idxmax()
                findings.append(
                    f"The highest fatal-collision rate among speed-limit groups with at least 100 records "
                    f"is {top} mph ({by_speed.loc[top, 'fatal_rate']:.2f} per 1,000 collisions)."
                )
        area = collisions["area_label"].value_counts()
        if not area.empty:
            findings.append(f"{area.index[0]} locations contribute the largest number of collisions in this selection.")
    if not casualties.empty:
        vru = casualties["is_vru"].mean() * 100
        findings.append(f"Vulnerable road users represent {vru:.1f}% of casualties in the current selection.")
        user_ksi = casualties.groupby("user_type").agg(
            casualties=("is_ksi", "size"), ksi=("is_ksi", "sum")
        )
        user_ksi = user_ksi[user_ksi["casualties"] >= 100]
        if not user_ksi.empty:
            user_ksi["ksi_share"] = user_ksi["ksi"] / user_ksi["casualties"] * 100
            group = user_ksi["ksi_share"].idxmax()
            findings.append(
                f"{group} has the highest killed-or-seriously-injured share among groups with at least 100 casualties "
                f"({user_ksi.loc[group, 'ksi_share']:.1f}%)."
            )
    return findings


def generate_recommendations(collisions, casualties):
    """Create cautious, descriptive prompts from the currently filtered data."""
    recommendations = []
    if not collisions.empty:
        fatal_share = collisions["is_fatal"].mean() * 100
        if fatal_share > 0:
            recommendations.append(("PRIORITY · SEVERITY", "Review fatal-collision locations",
                f"Fatal collisions account for {fatal_share:.2f}% of this selection. Examine the locations and road contexts associated with these records before proposing targeted interventions.", RED))
        dark = collisions["is_dark"].mean() * 100
        if dark >= 20:
            recommendations.append(("ROAD ENVIRONMENT", "Investigate night-time conditions",
                f"{dark:.1f}% of selected collisions occurred in darkness. Compare lighting conditions, road type and severity to identify where further local review may be useful.", AMBER))
        if "speed_label" in collisions and collisions["speed_label"].notna().any():
            speed = collisions[collisions["speed_label"] != "Unknown"].groupby("speed_label").agg(total=("is_fatal", "size"), fatal=("is_fatal", "sum"))
            speed = speed[speed["total"] >= 100]
            if not speed.empty:
                speed["rate"] = speed["fatal"] / speed["total"] * 1000
                top = speed["rate"].idxmax()
                recommendations.append(("SPEED CONTEXT", "Prioritise a closer speed-limit review",
                    f"The {top} mph group has the highest fatal-collision rate in this selection ({speed.loc[top, 'rate']:.2f} per 1,000 records, among groups with at least 100 collisions). This is an association, not proof of causation.", CORAL))
    if not casualties.empty:
        vru_share = casualties["is_vru"].mean() * 100
        if vru_share >= 25:
            recommendations.append(("ROAD USER PROTECTION", "Examine vulnerable-user exposure",
                f"Vulnerable road users represent {vru_share:.1f}% of casualties. Review pedestrian, cyclist and motorcyclist patterns separately when considering infrastructure or safety measures.", CYAN))
    if not recommendations:
        recommendations.append(("DATA REVIEW", "Broaden the current selection",
            "The active filters do not expose a strong descriptive signal for a recommendation. Expand the years or severity groups and compare the resulting patterns.", PURPLE))
    return recommendations[:4]


# =============================================================================
# 6. SIDEBAR, FILTERS AND GLOBAL HEADER
# =============================================================================

try:
    collisions_all, casualties_all, vehicles_all = load_stats19_data()
except Exception as exc:
    st.error("The Stats19 datasets could not be loaded.")
    st.exception(exc)
    st.markdown(
        "**Check:** place the collision, casualty and vehicle CSV files in the "
        "`data/` directory beside `App_Prediction.py`. Their filenames must contain "
        "`collision`, `casualt` and `vehicle` respectively. If you have separate "
        "annual exports, keep all matching years in that folder; the dashboard combines them."
    )
    st.stop()

valid_dates = collisions_all["date"].dropna()
if valid_dates.empty:
    st.error("No valid collision dates were found. Check the date column and CSV format.")
    st.stop()

with st.sidebar:
    st.markdown(
        f'<div style="padding:12px 3px 18px;border-bottom:1px solid {BORDER};margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;gap:10px;font-family:Space Grotesk;font-size:1.24rem;font-weight:800;color:{INK};letter-spacing:-.04em">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:11px;background:linear-gradient(145deg,{TEAL},{CYAN});color:#10202B;font-size:1.2rem">🚦</span>'
        f'<span>ROAD<span style="color:{TEAL}">/</span>SAFETY</span></div>'
        f'<div style="color:{MUTED};font-size:.65rem;letter-spacing:.16em;text-transform:uppercase;margin-top:3px">'
        f'Intelligence platform · STATS19</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-kicker'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio(
        "Dashboard section",
        ["Overview", "Collision Analytics", "Road Users",
         "Severity Prediction", "Collision Explorer", "Model Insights"],
        label_visibility="collapsed", key="main_navigation",
    )
    st.divider()
    st.markdown("<div class='section-kicker'>Refine the data</div>", unsafe_allow_html=True)
    years = sorted(valid_dates.dt.year.unique().tolist())
    severities = [x for x in SEVERITY_ORDER if x in collisions_all["collision_severity_label"].unique()]
    areas = sorted(collisions_all["area_label"].dropna().astype(str).unique().tolist())
    road_types = sorted(collisions_all["road_type_label"].dropna().astype(str).unique().tolist())
    speed_values = collisions_all["speed_label"].dropna().astype(str).unique().tolist()
    speed_values = sorted(speed_values, key=lambda x: (x == "Unknown", int(x) if x.isdigit() else 999))

    def reset_global_filters():
        for key in ["filter_years", "filter_severity", "filter_area", "filter_road", "filter_speed"]:
            st.session_state.pop(key, None)

    st.button("↺  Reset all filters", use_container_width=True, on_click=reset_global_filters)
    with st.expander("Time period & outcome", expanded=True):
        selected_years = st.multiselect(
            "Collision year", options=years, default=years, key="filter_years",
            placeholder="Choose one or more years",
            help="All years present in the loaded collision file are listed here. Select multiple years to compare them; monthly trends use a distinct colour for each year."
        )
        st.caption("Years available in loaded data: " + ", ".join(map(str, years)))
        selected_severity = st.multiselect(
            "Collision severity", severities, default=severities, key="filter_severity",
            placeholder="All severity levels"
        )
    with st.expander("Road context", expanded=False):
        selected_areas = st.multiselect("Area type", areas, default=areas, key="filter_area", placeholder="All area types")
        selected_roads = st.multiselect("Road type", road_types, default=road_types, key="filter_road", placeholder="All road types")
        selected_speeds = st.multiselect("Speed limit (mph)", speed_values, default=speed_values, key="filter_speed", placeholder="All speed limits")
    st.markdown(
        f'<div class="filter-caption">Filters apply across the analytics, road-user and collision-explorer sections. '
        f'Prediction inputs remain independent because they describe a hypothetical scenario.</div>',
        unsafe_allow_html=True,
    )

filtered_collisions = collisions_all.copy()
filtered_collisions = filtered_collisions[
    filtered_collisions["date"].dt.year.isin(selected_years)
    & filtered_collisions["collision_severity_label"].isin(selected_severity)
    & filtered_collisions["area_label"].astype(str).isin(selected_areas)
    & filtered_collisions["road_type_label"].astype(str).isin(selected_roads)
    & filtered_collisions["speed_label"].astype(str).isin(selected_speeds)
].copy()
filtered_casualties, filtered_vehicles = filter_linked_data(
    filtered_collisions, casualties_all, vehicles_all
)

if filtered_collisions.empty:
    st.warning("No collision records match the current global filters. Adjust the filters in the sidebar.")
    st.stop()

date_min = filtered_collisions["date"].min()
date_max = filtered_collisions["date"].max()
st.markdown(
    f'<div class="hero"><div class="eyebrow">UK DEPARTMENT FOR TRANSPORT · STATS19</div>'
    f'<div class="hero-title">Road Safety Intelligence</div>'
    f'<div class="hero-sub">A data-led view of UK road collisions, road-user outcomes and collision severity. '
    f'Explore patterns across {date_min:%Y}–{date_max:%Y}, investigate individual records, '
    f'and experiment with a research-only machine-learning model.</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="notice"><strong>RESEARCH-ONLY MODEL.</strong> {RESEARCH_ONLY_NOTICE}</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Current selection: **{len(filtered_collisions):,} collisions** · "
    f"**{len(filtered_casualties):,} casualties** · "
    f"**{len(filtered_vehicles):,} vehicles** · "
    f"{date_min:%d %b %Y} – {date_max:%d %b %Y}"
)


# =============================================================================
# 7. DASHBOARD SECTIONS
# =============================================================================

def render_overview():
    section_heading(
        "01 / OVERVIEW", "The road safety picture",
        "A high-level view of collision volume, severity, timing and location in the selected period.",
    )
    render_kpis(filtered_collisions, filtered_casualties)

    left, right = st.columns([1.05, 1.45], gap="large")
    with left:
        severity_donut(filtered_collisions, "collision_severity_label")
    with right:
        panel("Collision trend over time", "Monthly collision counts across the selected years.")
        trend = monthly_counts(filtered_collisions)
        if trend.empty:
            empty_state("No valid dates in the current selection.")
        else:
            chart = alt.Chart(trend).mark_line(
                strokeWidth=3, point=alt.OverlayMarkDef(size=48, filled=True)
            ).encode(
                x=alt.X("MonthNo:O", title=None, sort=list(range(1, 13)),
                        axis=alt.Axis(labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]")),
                y=alt.Y("Collisions:Q", title="Collisions", scale=alt.Scale(zero=True)),
                color=alt.Color("Year:N", title="Year", scale=alt.Scale(range=YEAR_PALETTE)),
                tooltip=[alt.Tooltip("Year:N"), alt.Tooltip("Month:N"), alt.Tooltip("Collisions:Q", format=",")],
            ).properties(height=390)
            show_chart(chart)

    left, right = st.columns([1, 1], gap="large")
    with left:
        area_counts = count_frame(filtered_collisions["area_label"], "Area", "Collisions")
        bar_chart(area_counts, "Area", "Collisions", "Collisions by area", colour=BLUE)
    with right:
        panel("Collision locations", "Recorded coordinates within the current selection; points are sampled for display.")
        geo = filtered_collisions.dropna(subset=["latitude", "longitude"]).copy()
        geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
        geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
        geo = geo.dropna(subset=["latitude", "longitude"])
        geo = geo[
            geo["latitude"].between(49, 61) & geo["longitude"].between(-9, 3)
        ]
        if geo.empty:
            empty_state("No valid UK coordinates are available for this selection.")
        else:
            if len(geo) > 5000:
                geo = geo.sample(5000, random_state=42)
            st.map(geo[["latitude", "longitude"]], latitude="latitude", longitude="longitude")

    st.markdown("### What stands out")
    findings = generate_key_findings(filtered_collisions, filtered_casualties)
    if findings:
        for item in findings[:5]:
            st.markdown(f"- {item}")
    else:
        st.info("There are not enough records to generate a reliable summary.")
    st.caption("Descriptive findings reflect the selected records; they do not establish causation.")

    st.markdown("### Suggested areas for attention")
    st.caption("Automatically generated from the active filters. These are investigation prompts, not policy or operational instructions.")
    recs = generate_recommendations(filtered_collisions, filtered_casualties)
    rec_cols = st.columns(min(2, len(recs)), gap="large")
    for idx, (tag, title, copy, colour) in enumerate(recs):
        with rec_cols[idx % len(rec_cols)]:
            st.markdown(
                f'<div class="recommendation-card"><div class="recommendation-tag" style="color:{colour}">{tag}</div>'
                f'<div class="recommendation-title">{title}</div><div class="recommendation-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )


def render_collision_analytics():
    section_heading(
        "02 / COLLISION ANALYTICS", "When and where collisions occur",
        "Explore temporal patterns and the road, lighting and environmental conditions recorded at collision scenes.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        panel("Collisions by weekday", "Count of collision records.")
        day_counts = count_frame(filtered_collisions["weekday"], "Weekday", "Collisions")
        day_counts["Weekday"] = pd.Categorical(day_counts["Weekday"], WEEKDAY_ORDER, ordered=True)
        day_counts = day_counts.sort_values("Weekday")
        bar_chart(day_counts, "Weekday", "Collisions", " ", colour=BLUE, order=WEEKDAY_ORDER)
    with c2:
        panel("Collisions by hour", "Hour is derived from the recorded collision time.")
        hour_counts = count_frame(
            filtered_collisions.loc[filtered_collisions["hour_label"] != "Unknown", "hour_label"],
            "Hour", "Collisions",
        )
        bar_chart(hour_counts, "Hour", "Collisions", " ", colour=TEAL, order=HOUR_ORDER)

    panel("Day and hour collision pattern", "Darker cells indicate more recorded collisions.")
    heat = (
        filtered_collisions[filtered_collisions["hour_label"] != "Unknown"]
        .groupby(["weekday", "hour_label"], observed=True).size()
        .reset_index(name="Collisions")
    )
    if heat.empty:
        empty_state("No valid day/hour observations.")
    else:
        chart = alt.Chart(heat).mark_rect(cornerRadius=3).encode(
            x=alt.X("hour_label:O", sort=HOUR_ORDER, title="Hour", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("weekday:O", sort=WEEKDAY_ORDER, title=None),
            color=alt.Color("Collisions:Q", scale=alt.Scale(scheme="tealblues"), title="Collisions"),
            tooltip=["weekday:N", "hour_label:N", alt.Tooltip("Collisions:Q", format=",")],
        ).properties(height=340)
        show_chart(chart)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        light = filtered_collisions.groupby(
            ["light_label", "collision_severity_label"], observed=True
        ).size().reset_index(name="Count")
        stacked_severity_chart(light, "light_label", title="Severity composition by light condition")
    with c2:
        road = filtered_collisions.groupby(
            ["road_type_label", "collision_severity_label"], observed=True
        ).size().reset_index(name="Count")
        stacked_severity_chart(road, "road_type_label", title="Severity composition by road type")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        weather = count_frame(filtered_collisions["weather_label"], "Weather", "Collisions")
        bar_chart(weather, "Weather", "Collisions", "Recorded weather conditions", colour=PURPLE, horizontal=True)
    with c2:
        surface = count_frame(filtered_collisions["surface_label"], "Surface", "Collisions")
        bar_chart(surface, "Surface", "Collisions", "Recorded road surface conditions", colour=AMBER, horizontal=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        speed = filtered_collisions[filtered_collisions["speed_label"] != "Unknown"].groupby(
            "speed_label", observed=True
        ).agg(
            Collisions=("is_fatal", "size"), Fatal=("is_fatal", "sum")
        ).reset_index()
        if not speed.empty:
            speed["Fatal per 1,000"] = speed["Fatal"] / speed["Collisions"] * 1000
            speed["speed_order"] = pd.to_numeric(speed["speed_label"], errors="coerce")
            speed = speed.sort_values("speed_order")
            panel("Fatal collisions per 1,000 records by speed limit", "Groups with fewer observations should be interpreted cautiously.")
            chart = alt.Chart(speed).mark_bar(color=RED, cornerRadiusEnd=4).encode(
                x=alt.X("speed_label:O", sort=speed["speed_label"].tolist(), title="Speed limit (mph)"),
                y=alt.Y("Fatal per 1,000:Q", title="Fatal collisions per 1,000"),
                tooltip=["speed_label:N", alt.Tooltip("Collisions:Q", format=","), alt.Tooltip("Fatal:Q", format=","), alt.Tooltip("Fatal per 1,000:Q", format=".2f")],
            ).properties(height=340)
            show_chart(chart)
        else:
            empty_state("No speed-limit records.")
    with c2:
        junction = count_frame(filtered_collisions["junction_detail_label"], "Junction", "Collisions")
        bar_chart(junction, "Junction", "Collisions", "Collisions by junction detail", colour=BLUE, horizontal=True)


def render_road_users():
    section_heading(
        "03 / ROAD USERS", "Who is affected",
        "Understand casualty severity, vulnerable road users, age patterns and the vehicle types involved.",
    )
    render_kpis(filtered_collisions, filtered_casualties)
    if filtered_casualties.empty:
        return empty_state("No casualty records are linked to the current collision selection.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        users = count_frame(filtered_casualties["user_type"], "Road user", "Casualties")
        bar_chart(users, "Road user", "Casualties", "Casualties by road-user group", colour=TEAL, horizontal=True)
    with c2:
        age = filtered_casualties.groupby(
            ["age_band_label", "casualty_severity_label"], observed=True
        ).size().reset_index(name="Casualties")
        stacked_severity_chart(age, "age_band_label", "Casualties", "Casualties by age band and severity")

    user_ksi = filtered_casualties.groupby("user_type").agg(
        Casualties=("is_ksi", "size"), KSI=("is_ksi", "sum")
    ).reset_index()
    user_ksi["KSI share (%)"] = user_ksi["KSI"] / user_ksi["Casualties"] * 100
    user_ksi = user_ksi[user_ksi["Casualties"] >= 50]
    panel("Killed or seriously injured share by road-user group", "KSI = casualties recorded as killed or seriously injured. Groups with fewer than 50 records are omitted.")
    if user_ksi.empty:
        empty_state("Not enough casualty records per road-user group for this comparison.")
    else:
        chart = alt.Chart(user_ksi).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("KSI share (%):Q", title="KSI casualties (%)"),
            y=alt.Y("user_type:N", sort="-x", title=None),
            color=alt.Color("user_type:N", scale=alt.Scale(scheme="tableau10"), legend=None),
            tooltip=["user_type:N", alt.Tooltip("Casualties:Q", format=","), alt.Tooltip("KSI:Q", format=","), alt.Tooltip("KSI share (%):Q", format=".1f")],
        ).properties(height=340)
        show_chart(chart)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        sex = count_frame(filtered_casualties["sex_label"], "Sex", "Casualties")
        panel("Casualties by recorded sex", "Recorded sex categories are shown separately.")
        if sex.empty:
            empty_state("No recorded sex values are available.")
        else:
            sex_colours = {"Male": BLUE, "Female": CORAL, "Not applicable": PURPLE, "Unknown": MUTED}
            sex_chart = alt.Chart(sex).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("Casualties:Q", title="Casualty records"),
                y=alt.Y("Sex:N", sort="-x", title=None),
                color=alt.Color("Sex:N", scale=alt.Scale(domain=list(sex_colours), range=list(sex_colours.values())), legend=None),
                tooltip=["Sex:N", alt.Tooltip("Casualties:Q", format=",")],
            ).properties(height=350)
            show_chart(sex_chart)
    with c2:
        vehicle_types = count_frame(filtered_vehicles["vehicle_type_label"], "Vehicle", "Vehicles")
        bar_chart(vehicle_types, "Vehicle", "Vehicles", "Vehicles involved", colour=AMBER, horizontal=True)

    severity = count_frame(filtered_casualties["casualty_severity_label"], "Severity", "Casualties")
    severity["sort_order"] = severity["Severity"].map({x: i for i, x in enumerate(SEVERITY_ORDER)})
    severity = severity.sort_values("sort_order")
    panel("Casualty totals by severity", "Counts of casualty records, not collision records. Fatal, serious and slight outcomes use consistent colours throughout the dashboard.")
    if severity.empty:
        empty_state("No casualty severity records are available.")
    else:
        severity_chart = alt.Chart(severity).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("Casualties:Q", title="Casualty records"),
            y=alt.Y("Severity:N", sort=SEVERITY_ORDER, title=None),
            color=alt.Color("Severity:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=[SEVERITY_COLOURS[x] for x in SEVERITY_ORDER]), legend=None),
            tooltip=["Severity:N", alt.Tooltip("Casualties:Q", format=",")],
        ).properties(height=350)
        show_chart(severity_chart)


# =============================================================================
# 8. CONTRACT-DRIVEN SEVERITY PREDICTION
# =============================================================================

MODEL_PATH = MODEL_DIR / "stats19_severity_model.joblib"
CONTRACT_PATH = MODEL_DIR / "model_contract.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

KEY_PREDICTORS = [
    "driver_age_mean", "vulnerable_road_user_count",
    "vulnerable_road_user_involved", "driver_age_min",
    "vehicle_type_first", "trunk_road_flag", "engine_cc_mean",
    "road_type", "time_band", "speed_limit",
]
PREDICTOR_LABELS = {
    "driver_age_mean": "Mean driver age",
    "vulnerable_road_user_count": "Vulnerable road-user count",
    "vulnerable_road_user_involved": "Vulnerable road user involved",
    "driver_age_min": "Youngest driver age",
    "vehicle_type_first": "Primary vehicle type",
    "trunk_road_flag": "Trunk road",
    "engine_cc_mean": "Mean engine capacity (cc)",
    "road_type": "Road type",
    "time_band": "Time of day",
    "speed_limit": "Speed limit (mph)",
}
ROAD_TYPE_INPUT = {
    "Roundabout": 1, "One-way street": 2, "Dual carriageway": 3,
    "Single carriageway": 6, "Slip road": 7, "Unknown": 9,
    "Missing / not recorded": -1,
}
BINARY_INPUT = {"No": 0, "Yes": 1}
TIME_BAND_INPUT = {
    "Night": 0, "Morning peak": 1, "Daytime": 2,
    "Evening peak": 3, "Late evening": 4, "Unknown": 5,
}
ADVANCED_CODE_LABELS = {
    "junction_detail": {
        **JUNCTION_DETAIL_LABELS,
        13: "T or staggered junction", 16: "Crossroads",
        17: "More than four arms", 18: "Private drive or entrance",
        19: "Other junction", 99: "Unknown junction detail",
    },
    "junction_control": JUNCTION_CONTROL_LABELS,
    "light_conditions": LIGHT_LABELS,
    "weather_conditions": WEATHER_LABELS,
    "road_surface_conditions": SURFACE_LABELS,
    "urban_or_rural_area": AREA_LABELS,
    "first_road_class": ROAD_CLASS_LABELS,
    "second_road_class": {0: "No second road", **ROAD_CLASS_LABELS},
    "special_conditions_at_site": {
        0: "None", 1: "Traffic signal out", 2: "Traffic signal defective",
        3: "Road sign/marking defective", 4: "Roadworks",
        5: "Road surface defective", 6: "Oil or diesel", 7: "Mud", -1: "Unknown",
    },
    "carriageway_hazards": {
        0: "None", 1: "Vehicle load on road", 2: "Other object on road",
        3: "Previous accident", 4: "Dog on road", 5: "Other animal on road",
        6: "Pedestrian in carriageway", 7: "Animal in carriageway", -1: "Unknown",
    },
    "driver_sex_first": SEX_LABELS,
    "vehicle_manoeuvre_first": {
        1: "Reversing", 2: "Parked", 3: "Waiting/held up",
        4: "Slowing/stopping", 5: "Moving off", 6: "U-turn",
        7: "Turning left", 8: "Waiting to turn left", 9: "Turning right",
        10: "Waiting to turn right", 11: "Changing lane left",
        12: "Changing lane right", 13: "Overtaking moving vehicle",
        14: "Overtaking stationary vehicle", 15: "Overtaking nearside",
        16: "Going ahead – left bend", 17: "Going ahead – right bend",
        18: "Going ahead – other", 99: "Unknown",
    },
}
BINARY_FEATURES = {
    "vulnerable_road_user_involved", "trunk_road_flag", "weekend_flag",
    "motorcycle_involved", "goods_vehicle_involved",
}


@st.cache_resource(show_spinner="Loading the saved severity model...")
def load_prediction_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Model contract not found: {CONTRACT_PATH}")
    model = joblib.load(MODEL_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(contract.get("features"), list) or not contract["features"]:
        raise ValueError("model_contract.json must contain a non-empty features list.")
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    specs = []
    for item in contract["features"]:
        if isinstance(item, str):
            specs.append({"feature": item, "dtype": "numeric", "default": 0})
        elif isinstance(item, dict):
            name = item.get("feature") or item.get("name")
            if not name:
                raise ValueError("Each model feature object must define 'feature' or 'name'.")
            specs.append({**item, "feature": name})
        else:
            raise ValueError("Unsupported model feature definition in model_contract.json.")
    names = [x["feature"] for x in specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names found in model_contract.json.")
    return model, contract, metadata, specs


def _numeric_default(spec):
    value = spec.get("default", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_numeric_spec(spec):
    return str(spec.get("dtype", "")).lower() in {"numeric", "number", "float", "integer", "int"}


def _contract_code_options(spec, mapping):
    codes = []
    for raw in spec.get("category_examples", []) or []:
        try:
            code = int(float(raw))
        except (TypeError, ValueError):
            continue
        if code in mapping and code not in codes:
            codes.append(code)
    if not codes:
        codes = list(mapping.keys())
    return codes


def render_prediction_input(spec):
    """Return a value in the representation expected by the saved model."""
    name = spec["feature"]
    label = PREDICTOR_LABELS.get(name, name.replace("_", " ").title())
    default = spec.get("default", 0)
    numeric = _is_numeric_spec(spec)
    key = f"prediction_{name}"

    def preserve_type(value):
        if numeric:
            try:
                return int(value) if float(value).is_integer() else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Input for {name} must be numeric.")
        return str(value)

    if name == "vehicle_type_first":
        mapping = VEHICLE_TYPE_LABELS
        codes = _contract_code_options(spec, mapping)
        labels = [mapping[c] for c in codes]
        try:
            default_code = int(float(default))
            index = codes.index(default_code) if default_code in codes else 0
        except (TypeError, ValueError):
            index = 0
        chosen = st.selectbox(label, labels, index=index, key=key)
        return preserve_type(codes[labels.index(chosen)])

    if name == "road_type":
        codes = _contract_code_options(spec, ROAD_TYPE_LABELS)
        labels = [ROAD_TYPE_LABELS[c] for c in codes]
        try:
            default_code = int(float(default))
            index = codes.index(default_code) if default_code in codes else 0
        except (TypeError, ValueError):
            index = 0
        chosen = st.selectbox(label, labels, index=index, key=key)
        return preserve_type(codes[labels.index(chosen)])

    if name in BINARY_FEATURES or name.endswith("_involved") or name.endswith("_flag"):
        try:
            default_code = int(float(default))
        except (TypeError, ValueError):
            default_code = 0
        chosen = st.selectbox(
            label, list(BINARY_INPUT), index=1 if default_code == 1 else 0, key=key
        )
        return preserve_type(BINARY_INPUT[chosen])

    if name == "time_band":
        codes = list(TIME_BAND_INPUT.values())
        labels = list(TIME_BAND_INPUT.keys())
        try:
            default_code = int(float(default))
            index = codes.index(default_code) if default_code in codes else 0
        except (TypeError, ValueError):
            index = 0
        chosen = st.selectbox(label, labels, index=index, key=key)
        # Critical: return the model's code, never the display string.
        return preserve_type(TIME_BAND_INPUT[chosen])

    if name in ADVANCED_CODE_LABELS:
        mapping = ADVANCED_CODE_LABELS[name]
        codes = _contract_code_options(spec, mapping)
        labels = [mapping[c] for c in codes]
        try:
            default_code = int(float(default))
            index = codes.index(default_code) if default_code in codes else 0
        except (TypeError, ValueError):
            index = 0
        chosen = st.selectbox(label, labels, index=index, key=key)
        return preserve_type(codes[labels.index(chosen)])

    if name in {"driver_age_mean", "driver_age_min"}:
        return st.number_input(
            label, min_value=0.0, max_value=100.0,
            value=float(_numeric_default(spec)), step=1.0, key=key,
        ) if numeric else st.text_input(label, str(default), key=key)

    if "engine_cc" in name:
        return st.number_input(
            label, min_value=0.0, max_value=20000.0,
            value=float(_numeric_default(spec)), step=100.0, key=key,
        ) if numeric else st.text_input(label, str(default), key=key)

    if name in {"speed_limit", "number_of_vehicles", "hour", "month", "day_of_week",
                "vulnerable_road_user_count", "vehicle_record_count"} and numeric:
        minimum = 0 if name not in {"month", "day_of_week"} else 1
        maximum = 100 if name in {"month", "day_of_week"} else 1000
        if name == "month":
            maximum = 12
        if name == "day_of_week":
            maximum = 7
        if name == "speed_limit":
            options = [20, 30, 40, 50, 60, 70]
            try:
                default_int = int(float(default))
                idx = options.index(default_int) if default_int in options else 1
            except (TypeError, ValueError):
                idx = 1
            return st.selectbox(label, options, index=idx, key=key)
        numeric_value = int(round(_numeric_default(spec)))
        numeric_value = min(max(numeric_value, minimum), maximum)
        return st.number_input(
            label, min_value=minimum, max_value=maximum,
            value=numeric_value, step=1, key=key,
        )

    if numeric:
        return st.number_input(
            label, value=float(_numeric_default(spec)), step=1.0, key=key
        )

    examples = [str(x) for x in (spec.get("category_examples") or []) if x is not None]
    default_text = str(default if default is not None else "Unknown")
    if default_text not in examples:
        examples.insert(0, default_text)
    examples = list(dict.fromkeys(examples)) or ["Unknown"]
    return st.selectbox(
        label, examples, index=examples.index(default_text), key=key
    )


def model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        for estimator in reversed(list(model.named_steps.values())):
            if hasattr(estimator, "classes_"):
                return list(estimator.classes_)
    if hasattr(model, "fitted_estimator") and hasattr(model.fitted_estimator, "classes_"):
        return list(model.fitted_estimator.classes_)
    return [1, 2, 3]


def severity_name(value):
    return SEVERITY_LABELS.get(value, SEVERITY_LABELS.get(str(value), str(value)))


def render_prediction():
    section_heading(
        "04 / SEVERITY PREDICTION", "Explore a collision scenario",
        "Enter a hypothetical collision profile and inspect the saved model's predicted class and class scores.",
    )
    st.markdown(
        f'<div class="notice"><strong>IMPORTANT LIMITATION.</strong> {RESEARCH_ONLY_NOTICE} '
        f'This model estimates severity conditional on a collision being recorded; it does not forecast collision occurrence.</div>',
        unsafe_allow_html=True,
    )
    try:
        model, contract, metadata, specs = load_prediction_assets()
    except ModuleNotFoundError as exc:
        if exc.name == "catboost" or str(exc.name).startswith("catboost."):
            st.error("The saved model needs CatBoost, but this Streamlit process cannot import it.")
            try:
                catboost_spec = importlib.util.find_spec("catboost")
            except (ImportError, ValueError):
                catboost_spec = None
            try:
                catboost_version = importlib_metadata.version("catboost")
            except importlib_metadata.PackageNotFoundError:
                catboost_version = "Not registered in this interpreter"
            st.markdown("**Runtime diagnostics**")
            st.code(
                f"Python: {sys.version.split()[0]}\n"
                f"Interpreter: {sys.executable}\n"
                f"CatBoost package metadata: {catboost_version}\n"
                f"CatBoost import location: {getattr(catboost_spec, 'origin', None) or 'Not found'}",
                language="text",
            )
            st.markdown("Run this diagnostic command in the same terminal used to launch Streamlit:")
            diagnostic_command = (
                f'& "{sys.executable}" -c "import sys; print(sys.version); '
                'print(sys.executable); import catboost; '
                'print(catboost.__version__); print(catboost.__file__)"'
            )
            st.code(diagnostic_command, language="powershell")
            st.caption("If the command fails, copy its complete output. If it succeeds, the package is importable and we should investigate the saved model's pickle dependencies rather than reinstalling CatBoost.")
            st.markdown("If CatBoost is genuinely missing, install or upgrade it with:")
            st.code(f'& "{sys.executable}" -m pip install --upgrade "catboost>=1.2.8"', language="powershell")
            if st.button("Clear model cache and retry", key="retry_prediction_assets"):
                load_prediction_assets.clear()
                st.rerun()
        else:
            st.error(f"A required Python module is missing: {exc.name}")
            st.exception(exc)
        return
    except Exception as exc:
        st.error("The prediction assets could not be loaded.")
        st.exception(exc)
        st.info("Confirm that the model, model_contract.json and metadata.json are in the models/ folder.")
        return

    names = [spec["feature"] for spec in specs]
    primary = [spec for spec in specs if spec["feature"] in KEY_PREDICTORS]
    advanced = [spec for spec in specs if spec["feature"] not in KEY_PREDICTORS]
    st.markdown(
        f"**Model input contract:** {len(specs)} features · "
        f"**Estimator:** {type(model).__name__}"
    )
    values = {}
    with st.form("collision_severity_prediction"):
        st.markdown("#### Main collision characteristics")
        cols = st.columns(2)
        for i, spec in enumerate(primary):
            with cols[i % 2]:
                values[spec["feature"]] = render_prediction_input(spec)
        if advanced:
            with st.expander(f"Additional model inputs ({len(advanced)} features)", expanded=False):
                cols = st.columns(2)
                for i, spec in enumerate(advanced):
                    with cols[i % 2]:
                        values[spec["feature"]] = render_prediction_input(spec)
        submitted = st.form_submit_button(
            "Run severity prediction", type="primary", use_container_width=True
        )

    if not submitted:
        st.caption("The form is generated from the saved model contract. Expand the additional inputs to review all model features.")
        return

    try:
        row = {}
        for spec in specs:
            name = spec["feature"]
            value = values[name]
            row[name] = value
        input_frame = pd.DataFrame([row], columns=names)

        # Do not reorder, rename or silently drop features: the contract is authoritative.
        if list(input_frame.columns) != names:
            raise ValueError("Prediction input columns do not match the saved feature contract.")
        prediction = np.asarray(model.predict(input_frame)).reshape(-1)[0]
        predicted_label = severity_name(prediction)

        probabilities = None
        classes = model_classes(model)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(input_frame))[0]
        elif hasattr(model, "fitted_estimator") and hasattr(model.fitted_estimator, "predict_proba"):
            probabilities = np.asarray(model.fitted_estimator.predict_proba(input_frame))[0]

        st.markdown("---")
        colour = SEVERITY_COLOURS.get(predicted_label, BLUE)
        st.markdown(
            f'<div class="result-card" style="max-width:620px;border-left:5px solid {colour}">'
            f'<div class="result-label">Predicted collision severity</div>'
            f'<div class="result-value" style="color:{colour}">{predicted_label}</div>'
            f'<div class="small-muted">Research model output · not an operational assessment</div></div>',
            unsafe_allow_html=True,
        )

        if probabilities is not None:
            if len(classes) != len(probabilities):
                st.warning("The estimator returned probabilities that could not be aligned to its class labels.")
            else:
                scores = pd.DataFrame({
                    "Severity": [severity_name(x) for x in classes],
                    "Model score": probabilities.astype(float),
                })
                scores["Score (%)"] = scores["Model score"] * 100
                scores["Severity"] = pd.Categorical(
                    scores["Severity"], categories=SEVERITY_ORDER, ordered=True
                )
                scores = scores.sort_values("Severity")
                panel("Class score breakdown", "Scores are the estimator's outputs; they are not guaranteed to be calibrated probabilities.")
                chart = alt.Chart(scores).mark_bar(cornerRadiusEnd=5).encode(
                    x=alt.X("Score (%):Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Severity:N", sort=SEVERITY_ORDER, title=None),
                    color=alt.Color(
                        "Severity:N",
                        scale=alt.Scale(domain=SEVERITY_ORDER,
                                        range=[SEVERITY_COLOURS[x] for x in SEVERITY_ORDER]),
                        legend=None,
                    ),
                    tooltip=["Severity:N", alt.Tooltip("Score (%):Q", format=".2f")],
                ).properties(height=340)
                show_chart(chart)
        else:
            st.info("This estimator does not expose predict_proba(); class scores are unavailable.")

        with st.expander("Review the exact inputs sent to the model"):
            st.dataframe(input_frame.T.rename(columns={0: "Input value"}), use_container_width=True)

        with st.expander("Prediction scope and limitations"):
            st.write(contract.get("scope_warning", "The model predicts collision severity conditional on a recorded collision."))
            st.warning(contract.get("operational_warning", RESEARCH_ONLY_NOTICE))
            st.write("The prediction is sensitive to the feature values and to the data distribution used during training.")
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.exception(exc)


# =============================================================================
# 9. COLLISION EXPLORER
# =============================================================================

def render_collision_explorer():
    section_heading(
        "05 / COLLISION EXPLORER", "Investigate individual collision records",
        "Search the filtered collision records and inspect linked vehicle and casualty information.",
    )
    records = filtered_collisions.copy()
    left, right = st.columns([2, 1])
    with left:
        query = st.text_input("Search collision ID, road type or area", key="explorer_search")
    with right:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=0, key="explorer_page_size")

    if query.strip():
        q = query.strip().lower()
        mask = (
            records["collision_index"].astype(str).str.lower().str.contains(q, na=False)
            | records["road_type_label"].astype(str).str.lower().str.contains(q, na=False)
            | records["area_label"].astype(str).str.lower().str.contains(q, na=False)
        )
        records = records[mask]

    severity_filter = st.multiselect(
        "Narrow by severity", SEVERITY_ORDER,
        default=[s for s in SEVERITY_ORDER if s in records["collision_severity_label"].unique()],
        key="explorer_severity",
    )
    records = records[records["collision_severity_label"].isin(severity_filter)]
    records = records.sort_values("date", ascending=False)
    if records.empty:
        return empty_state("No records match the explorer search and filters.")

    total_pages = max(1, int(np.ceil(len(records) / page_size)))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page) - 1) * page_size
    current = records.iloc[start:start + page_size].copy()

    display_columns = [
        "collision_index", "date", "time", "collision_severity_label",
        "number_of_vehicles", "number_of_casualties", "speed_label",
        "road_type_label", "area_label", "light_label",
    ]
    display_columns = [c for c in display_columns if c in current.columns]
    display = current[display_columns].rename(columns={
        "collision_index": "Collision ID", "date": "Date", "time": "Time",
        "collision_severity_label": "Severity", "number_of_vehicles": "Vehicles",
        "number_of_casualties": "Casualties", "speed_label": "Speed limit (mph)",
        "road_type_label": "Road type", "area_label": "Area", "light_label": "Lighting",
    })
    st.caption(f"Showing {len(current):,} of {len(records):,} matching collision records.")
    st.dataframe(display, hide_index=True, use_container_width=True)

    collision_ids = current["collision_index"].astype(str).tolist()
    selected_id = st.selectbox("Select a collision to inspect", collision_ids, key="explorer_collision_id")
    if selected_id:
        record = records[records["collision_index"].astype(str).eq(selected_id)].iloc[0]
        st.markdown(f"### Collision {selected_id}")
        date_text = record["date"].strftime("%d %b %Y") if pd.notna(record["date"]) else "Date unavailable"
        st.markdown(
            f"**{record['collision_severity_label']}** · {date_text} · "
            f"{record['area_label']} · {record['road_type_label']} · "
            f"{record['speed_label']} mph · {record['light_label']}"
        )
        v1, v2 = st.columns(2, gap="large")
        with v1:
            panel("Vehicles involved")
            linked_vehicles = filtered_vehicles[
                filtered_vehicles["collision_index"].astype(str).eq(selected_id)
            ]
            vehicle_cols = [
                c for c in ["vehicle_type_label", "driver_age_band_label",
                            "driver_sex_label", "vehicle_manoeuvre_label", "engine_capacity_cc"]
                if c in linked_vehicles.columns
            ]
            if linked_vehicles.empty:
                empty_state("No linked vehicle records were found.")
            else:
                st.dataframe(linked_vehicles[vehicle_cols], hide_index=True, use_container_width=True)
        with v2:
            panel("Casualties involved")
            linked_casualties = filtered_casualties[
                filtered_casualties["collision_index"].astype(str).eq(selected_id)
            ]
            casualty_cols = [
                c for c in ["user_type", "age_band_label", "sex_label",
                            "casualty_severity_label", "casualty_class_label"]
                if c in linked_casualties.columns
            ]
            if linked_casualties.empty:
                empty_state("No linked casualty records were found.")
            else:
                st.dataframe(linked_casualties[casualty_cols], hide_index=True, use_container_width=True)

    st.markdown("""
    <style>
    /* Download button styling */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #14b8a6, #0f766e);
        color: white;
        border: 1px solid rgba(20, 184, 166, 0.4);
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        transition: all 0.25s ease-in-out;
    }

    /* Hover effect */
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #2dd4bf, #0d9488);
        color: white;
        border-color: #5eead4;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(20, 184, 166, 0.25);
    }

    /* Click effect */
    div[data-testid="stDownloadButton"] button:active {
        transform: translateY(0);
        box-shadow: none;
    }
    </style>
    """, unsafe_allow_html=True)

    st.download_button(
        "Download matching collision records",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="uk_collision_records_page.csv",
        mime="text/csv",
    )


# =============================================================================
# 10. MODEL INSIGHTS
# =============================================================================

def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _normalise_key(key):
    return str(key).lower().replace("-", "_").replace(" ", "_")


def find_metric(metadata: dict, aliases: list[str]):
    """Find only explicitly named metric keys; never infer a metric from an arbitrary number."""
    wanted = {_normalise_key(x) for x in aliases}
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if _normalise_key(key) in wanted and isinstance(value, (int, float)):
                    return float(value)
            for value in obj.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found is not None:
                    return found
        return None
    return walk(metadata)


def final_estimator(model):
    if hasattr(model, "fitted_estimator"):
        return final_estimator(model.fitted_estimator)
    if hasattr(model, "named_steps"):
        return final_estimator(list(model.named_steps.values())[-1])
    if hasattr(model, "model_"):
        return model.model_
    return model


def render_model_insights():
    section_heading(
        "06 / MODEL INSIGHTS", "How the severity model performs",
        "Review the saved evaluation evidence, model characteristics and limitations before interpreting any prediction.",
    )
    st.markdown(
        f'<div class="notice"><strong>NOT APPROVED FOR OPERATIONAL USE.</strong> {RESEARCH_ONLY_NOTICE}</div>',
        unsafe_allow_html=True,
    )
    metadata = _load_json(METADATA_PATH)
    contract = _load_json(CONTRACT_PATH)
    if not metadata:
        st.warning("metadata.json is missing or could not be read. Model evaluation metrics cannot be displayed.")
    else:
        status = metadata.get("use_status") or metadata.get("acceptance_decision")
        if status:
            st.markdown(f"**Recorded model status:** {status}")

        metric_specs = [
            ("Macro F1", ["macro_f1", "f1_macro", "test_macro_f1"]),
            ("Balanced accuracy", ["balanced_accuracy", "test_balanced_accuracy"]),
            ("Fatal precision", ["fatal_precision"]),
            ("Fatal recall", ["fatal_recall"]),
            ("Serious precision", ["serious_precision"]),
            ("Serious recall", ["serious_recall"]),
        ]
        available = []
        for label, aliases in metric_specs:
            value = find_metric(metadata, aliases)
            if value is not None:
                available.append((label, value))
        if available:
            cols = st.columns(min(3, len(available)))
            for i, (label, value) in enumerate(available):
                with cols[i % len(cols)]:
                    st.metric(label, f"{value:.3f}")
        else:
            st.info("No recognised evaluation metrics were found in metadata.json. No values have been inferred.")

    st.markdown("### Saved model artefacts")
    model_path_exists = MODEL_PATH.exists()
    contract_path_exists = CONTRACT_PATH.exists()
    metadata_path_exists = METADATA_PATH.exists()
    status_rows = pd.DataFrame([
        {"Artefact": "Trained estimator", "File": str(MODEL_PATH.relative_to(APP_DIR)), "Available": "Yes" if model_path_exists else "No"},
        {"Artefact": "Feature contract", "File": str(CONTRACT_PATH.relative_to(APP_DIR)), "Available": "Yes" if contract_path_exists else "No"},
        {"Artefact": "Evaluation metadata", "File": str(METADATA_PATH.relative_to(APP_DIR)), "Available": "Yes" if metadata_path_exists else "No"},
    ])
    st.dataframe(status_rows, hide_index=True, use_container_width=True)

    try:
        model, _, _, specs = load_prediction_assets()
        estimator = final_estimator(model)
        importances = None
        feature_names = [s["feature"] for s in specs]
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_).ravel()
            if len(values) == len(feature_names):
                importances = pd.DataFrame({"Feature": feature_names, "Importance": values})
        elif hasattr(estimator, "coef_"):
            values = np.asarray(estimator.coef_)
            if values.ndim == 2:
                values = np.mean(np.abs(values), axis=0)
            if len(values) == len(feature_names):
                importances = pd.DataFrame({"Feature": feature_names, "Importance": values})
        if importances is not None:
            importances = importances.sort_values("Importance", ascending=False).head(20)
            panel("Top model feature importance", "Estimator-derived importance; this describes model reliance, not causation.")
            chart = alt.Chart(importances).mark_bar(color=TEAL, cornerRadiusEnd=4).encode(
                x=alt.X("Importance:Q", title="Importance"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
            ).properties(height=420)
            show_chart(chart)
        else:
            st.info("Feature importance is not exposed by the saved estimator in a directly alignable form.")
    except Exception as exc:
        st.warning(f"Feature importance could not be extracted from the saved estimator: {exc}")

    # Display a confusion matrix only when a real saved artefact is present.
    confusion_candidates = [
        OUTPUT_DIR / "confusion_matrix.csv",
        OUTPUT_DIR / "confusion_matrix.json",
        MODEL_DIR / "confusion_matrix.csv",
        MODEL_DIR / "confusion_matrix.json",
    ]
    confusion = None
    for artifact in confusion_candidates:
        if not artifact.exists():
            continue
        try:
            if artifact.suffix.lower() == ".csv":
                confusion = pd.read_csv(artifact, index_col=0)
            else:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                matrix = payload.get("confusion_matrix", payload.get("matrix", payload))
                labels = payload.get("labels", SEVERITY_ORDER) if isinstance(payload, dict) else SEVERITY_ORDER
                confusion = pd.DataFrame(matrix, index=labels, columns=labels)
            if confusion is not None and not confusion.empty:
                break
        except (ValueError, OSError, json.JSONDecodeError, TypeError):
            confusion = None
    if confusion is not None and not confusion.empty:
        panel("Saved test-set confusion matrix", "Only shown when a confusion-matrix artefact exists in outputs/ or models/.")
        confusion.index.name = "Actual"
        confusion.columns.name = "Predicted"
        st.dataframe(confusion, use_container_width=True)

    st.markdown("### Interpretation and limitations")
    limitations = [
        "The target is collision severity for recorded personal-injury collisions; the model does not predict collision occurrence.",
        "Fatal outcomes are a small minority class, so class imbalance materially affects precision and overall accuracy.",
        "The model is evaluated on a later chronological period; performance may change as reporting practices, road conditions and traffic patterns change.",
        "Feature importance is not evidence that a feature causes a collision or a more severe outcome.",
        "Predicted scores should not be interpreted as calibrated probabilities unless calibration has been separately evaluated.",
        "The model has not passed the project's acceptance gate and must remain a research demonstration.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")

    if contract.get("scope_warning"):
        st.info(contract["scope_warning"])
    if contract.get("operational_warning"):
        st.warning(contract["operational_warning"])


# =============================================================================
# 11. ROUTING AND FOOTER
# =============================================================================

if page == "Overview":
    render_overview()
elif page == "Collision Analytics":
    render_collision_analytics()
elif page == "Road Users":
    render_road_users()
elif page == "Severity Prediction":
    render_prediction()
elif page == "Collision Explorer":
    render_collision_explorer()
elif page == "Model Insights":
    render_model_insights()

st.markdown(
    f'<div class="footer"><strong style="color:{INK}">ROAD SAFETY INTELLIGENCE</strong>'
    f' · UK Department for Transport STATS19 collision, vehicle and casualty records'
    f'<br>Analytical and educational project. Model outputs are research-only and not intended for operational decisions.</div>',
    unsafe_allow_html=True,
)
