"""Load and consistently label the three linked Stats19 datasets."""
from pathlib import Path
import pandas as pd
import streamlit as st
from road_safety.config import (
    APP_DIR, DATA_DIR, COLLISION_COLS, CASUALTY_COLS, VEHICLE_COLS,
    SEVERITY_LABELS, AREA_LABELS, ROAD_TYPE_LABELS, LIGHT_LABELS,
    WEATHER_LABELS, SURFACE_LABELS, JUNCTION_DETAIL_LABELS,
    JUNCTION_CONTROL_LABELS, ROAD_CLASS_LABELS, SEX_LABELS,
    CASUALTY_CLASS_LABELS, AGE_BAND_LABELS, VEHICLE_TYPE_LABELS,
)

SEARCH_DIRS = [DATA_DIR, APP_DIR]

def find_file(keyword: str) -> Path:
    matches = []
    for folder in SEARCH_DIRS:
        if folder.exists():
            matches.extend(p for p in folder.glob("*.csv") if keyword.lower() in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"Could not find a CSV containing '{keyword}' in {DATA_DIR} or {APP_DIR}.")
    # Prefer the official raw Stats19 file when duplicate exports exist.
    return sorted(matches, key=lambda p: ("clean" in p.name.lower(), len(p.name)))[0]

def _read_available(path: Path, wanted: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    selected = [c for c in wanted if c in header]
    return pd.read_csv(path, usecols=selected, low_memory=False)

def _code_label(series: pd.Series, mapping: dict, unknown: str = "Unknown") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(mapping)
    # Retain meaningful source strings where a code mapping is not applicable.
    return mapped.fillna(series.astype("string").replace({"nan": pd.NA, "-1": "Data missing"})).fillna(unknown).astype("string")

def _normalise_id(frame: pd.DataFrame) -> pd.DataFrame:
    if "collision_index" in frame:
        frame["collision_index"] = frame["collision_index"].astype("string").str.strip()
    return frame

@st.cache_data(show_spinner="Reading Stats19 collision, casualty and vehicle records...")
def load_data():
    collision_path = find_file("collision")
    casualty_path = find_file("casualty")
    vehicle_path = find_file("vehicle")
    collisions = _normalise_id(_read_available(collision_path, COLLISION_COLS))
    casualties = _normalise_id(_read_available(casualty_path, CASUALTY_COLS))
    vehicles = _normalise_id(_read_available(vehicle_path, VEHICLE_COLS))

    if "date" in collisions:
        collisions["date"] = pd.to_datetime(collisions["date"], errors="coerce", dayfirst=True)
    if "time" in collisions:
        collisions["time"] = collisions["time"].astype("string").str.strip()
        parsed_time = pd.to_datetime(collisions["time"], format="%H:%M", errors="coerce")
        collisions["hour"] = parsed_time.dt.hour
        collisions["hour_label"] = collisions["hour"].map(lambda x: f"{int(x):02d}:00" if pd.notna(x) else "Unknown")
    else:
        collisions["hour"] = pd.NA
        collisions["hour_label"] = "Unknown"

    mappings = {
        "collision_severity": ("collision_severity_label", SEVERITY_LABELS),
        "urban_or_rural_area": ("area_label", AREA_LABELS),
        "road_type": ("road_type_label", ROAD_TYPE_LABELS),
        "light_conditions": ("light_label", LIGHT_LABELS),
        "weather_conditions": ("weather_conditions_label", WEATHER_LABELS),
        "road_surface_conditions": ("road_surface_conditions_label", SURFACE_LABELS),
        "junction_detail": ("junction_detail_label", JUNCTION_DETAIL_LABELS),
        "junction_control": ("junction_control_label", JUNCTION_CONTROL_LABELS),
        "first_road_class": ("first_road_class_label", ROAD_CLASS_LABELS),
        "second_road_class": ("second_road_class_label", ROAD_CLASS_LABELS),
    }
    for source, (target, mapping) in mappings.items():
        if source in collisions:
            collisions[target] = _code_label(collisions[source], mapping)
    aliases = {
        "urban_or_rural_label": "area_label",
        "light_conditions_label": "light_label",
    }
    for alias, source in aliases.items():
        if source in collisions:
            collisions[alias] = collisions[source]
    if "date" in collisions:
        collisions["weekday"] = collisions["date"].dt.day_name().fillna("Unknown")
        collisions["month_label"] = collisions["date"].dt.month_name().fillna("Unknown")
        collisions["month"] = collisions["date"].dt.month
        collisions["is_weekend"] = collisions["date"].dt.dayofweek.ge(5).fillna(False)
    else:
        collisions["weekday"] = "Unknown"
        collisions["month_label"] = "Unknown"
        collisions["month"] = pd.NA
        collisions["is_weekend"] = False
    if "collision_severity" in collisions:
        severity_num = pd.to_numeric(collisions["collision_severity"], errors="coerce")
        collisions["is_fatal"] = severity_num.eq(1)
    else:
        collisions["is_fatal"] = False
    collisions["is_dark"] = pd.to_numeric(collisions.get("light_conditions", pd.Series(index=collisions.index)), errors="coerce").isin([4, 5, 6, 7])
    if "speed_limit" in collisions:
        speed = pd.to_numeric(collisions["speed_limit"], errors="coerce")
        collisions["speed_label"] = speed.map(lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "Unknown")

    if "casualty_severity" in casualties:
        casualties["casualty_severity_label"] = _code_label(casualties["casualty_severity"], SEVERITY_LABELS)
        casualty_sev = pd.to_numeric(casualties["casualty_severity"], errors="coerce")
        casualties["is_ksi"] = casualty_sev.isin([1, 2])
    else:
        casualties["casualty_severity_label"] = "Unknown"
        casualties["is_ksi"] = False
    if "age_band_of_casualty" in casualties:
        casualties["age_band_label"] = _code_label(casualties["age_band_of_casualty"], AGE_BAND_LABELS)
    if "sex_of_casualty" in casualties:
        casualties["sex_label"] = _code_label(casualties["sex_of_casualty"], SEX_LABELS)
    if "casualty_class" in casualties:
        casualties["casualty_class_label"] = _code_label(casualties["casualty_class"], CASUALTY_CLASS_LABELS)
    if "casualty_type" in casualties:
        casualties["casualty_type_label"] = _code_label(casualties["casualty_type"], VEHICLE_TYPE_LABELS)
        ctype = pd.to_numeric(casualties["casualty_type"], errors="coerce")
        cclass = pd.to_numeric(casualties.get("casualty_class", pd.Series(index=casualties.index)), errors="coerce")
        user_type = pd.Series("Other", index=casualties.index, dtype="string")
        user_type.loc[ctype.eq(1) | cclass.eq(3)] = "Pedestrian"
        user_type.loc[ctype.eq(2)] = "Cyclist"
        user_type.loc[ctype.isin([3, 4, 5, 23])] = "Motorcyclist"
        user_type.loc[ctype.eq(9)] = "Car occupant"
        user_type.loc[ctype.isna() | ctype.eq(-1)] = "Unknown"
        casualties["user_type"] = user_type
    else:
        casualties["user_type"] = "Unknown"
    casualties["is_vru"] = casualties["user_type"].isin(["Pedestrian", "Cyclist", "Motorcyclist"])

    if "sex_of_driver" in vehicles:
        vehicles["driver_sex_label"] = _code_label(vehicles["sex_of_driver"], SEX_LABELS)
    if "age_band_of_driver" in vehicles:
        vehicles["driver_age_band_label"] = _code_label(vehicles["age_band_of_driver"], AGE_BAND_LABELS)
    if "vehicle_type" in vehicles:
        vehicles["vehicle_type_label"] = _code_label(vehicles["vehicle_type"], VEHICLE_TYPE_LABELS)
    if "vehicle_manoeuvre" in vehicles:
        manoeuvre_map = {1:"Reversing",2:"Parked",3:"Waiting to go",4:"Slowing or stopping",5:"Moving off",6:"U-turn",7:"Turning left",8:"Waiting to turn left",9:"Turning right",10:"Waiting to turn right",11:"Changing lane left",12:"Changing lane right",13:"Overtaking",14:"Going ahead left-hand bend",15:"Going ahead right-hand bend",16:"Going ahead other",17:"Not known",18:"Other"}
        vehicles["vehicle_manoeuvre_label"] = _code_label(vehicles["vehicle_manoeuvre"], manoeuvre_map)
    return collisions, casualties, vehicles

def get_baseline_metrics(collisions: pd.DataFrame, casualties: pd.DataFrame) -> dict:
    sev = collisions.get("collision_severity_label", pd.Series(dtype="string"))
    casualty_sev = casualties.get("casualty_severity_label", pd.Series(dtype="string"))
    total = len(collisions)
    return {
        "collisions": total,
        "casualties": len(casualties),
        "fatal_collisions": int(sev.eq("Fatal").sum()),
        "serious_collisions": int(sev.eq("Serious").sum()),
        "slight_collisions": int(sev.eq("Slight").sum()),
        "killed": int(casualty_sev.eq("Fatal").sum()),
        "seriously_injured": int(casualty_sev.eq("Serious").sum()),
        "vru_share": float(casualties.get("is_vru", pd.Series(dtype=bool)).mean() * 100) if len(casualties) else 0.0,
        "dark_share": float(collisions.get("is_dark", pd.Series(dtype=bool)).mean() * 100) if total else 0.0,
    }
