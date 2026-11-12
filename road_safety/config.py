"""Shared paths, labels, colour palette and dashboard settings."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

INK = "#14213D"
BLUE = "#3977D5"
TEAL = "#0F9D91"
AMBER = "#E6A23C"
RED = "#D94F5C"
GREEN = "#27966F"
GREY = "#94A3B8"
SLATE = "#64748B"
BACKGROUND = "#F4F7FB"
WHITE = "#FFFFFF"

SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLOURS = {"Fatal": RED, "Serious": AMBER, "Slight": TEAL}
SEVERITY_COLORS = SEVERITY_COLOURS
AREA_ORDER = ["Urban", "Rural", "Unclassified"]
AREA_COLOURS = {"Urban": BLUE, "Rural": TEAL, "Unclassified": GREY}

COLLISION_COLS = [
    "collision_index", "date", "time", "day_of_week", "collision_severity",
    "urban_or_rural_area", "speed_limit", "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions", "junction_detail",
    "junction_control", "first_road_class", "second_road_class",
    "special_conditions_at_site", "carriageway_hazards", "latitude", "longitude",
    "number_of_vehicles", "number_of_casualties", "pedestrian_crossing_human_control",
    "pedestrian_crossing_physical_facilities", "did_police_officer_attend_scene_of_accident",
    "local_authority_district", "local_authority_ons_district", "local_authority_highway",
    "lsoa_of_accident_location", "first_road_number", "second_road_number",
]
CASUALTY_COLS = ["collision_index", "casualty_severity", "casualty_class", "age_band_of_casualty", "sex_of_casualty", "casualty_type"]
VEHICLE_COLS = ["collision_index", "sex_of_driver", "age_band_of_driver", "vehicle_type", "vehicle_manoeuvre", "engine_capacity_cc"]

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight", "1": "Fatal", "2": "Serious", "3": "Slight"}
SEVERITY = SEVERITY_LABELS
AREA_LABELS = {1: "Urban", 2: "Rural", 3: "Unclassified", "1": "Urban", "2": "Rural", "3": "Unclassified"}
ROAD_TYPE_LABELS = {1: "Roundabout", 2: "One-way street", 3: "Dual carriageway", 6: "Single carriageway", 7: "Slip road", 9: "Unknown", 12: "One-way street / roundabout"}
LIGHT_LABELS = {1: "Daylight", 4: "Darkness – lights lit", 5: "Darkness – lights unlit", 6: "Darkness – no lighting", 7: "Darkness – lighting unknown", -1: "Data missing"}
WEATHER_LABELS = {1: "Fine, no high winds", 2: "Raining, no high winds", 3: "Snowing, no high winds", 4: "Fine, high winds", 5: "Raining, high winds", 6: "Snowing, high winds", 7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Data missing"}
SURFACE_LABELS = {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood over 3cm", 6: "Oil or diesel", 7: "Mud", 9: "Unknown", -1: "Data missing"}
JUNCTION_DETAIL_LABELS = {0: "Not at junction", 1: "Roundabout", 2: "Mini-roundabout", 3: "T-junction", 5: "Slip road", 6: "Crossroads", 7: "More than four arms", 8: "Private drive or entrance", 9: "Other junction", -1: "Data missing"}
JUNCTION_CONTROL_LABELS = {0: "Not at junction", 1: "Authorised person", 2: "Auto traffic signal", 3: "Stop sign", 4: "Give way or uncontrolled", -1: "Data missing"}
ROAD_CLASS_LABELS = {1: "Motorway", 2: "A(M)", 3: "A road", 4: "B road", 5: "C road", 6: "Unclassified", -1: "Data missing"}
SEX_LABELS = {1: "Male", 2: "Female", 3: "Unknown", -1: "Data missing"}
CASUALTY_CLASS_LABELS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian", -1: "Data missing"}
AGE_BAND_LABELS = {1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20", 5: "21–25", 6: "26–35", 7: "36–45", 8: "46–55", 9: "56–65", 10: "66–75", 11: "Over 75", -1: "Data missing"}
AGE_BAND_ORDER = ["0–5", "6–10", "11–15", "16–20", "21–25", "26–35", "36–45", "46–55", "56–65", "66–75", "Over 75"]
VEHICLE_TYPE_LABELS = {1: "Pedal cycle", 2: "Motorcycle 50cc and under", 3: "Motorcycle 125cc and under", 4: "Motorcycle over 125cc", 5: "Motorcycle over 500cc", 8: "Taxi or private hire", 9: "Car", 10: "Minibus", 11: "Bus or coach", 16: "Horse", 17: "Agricultural vehicle", 18: "Tram", 19: "Van", 20: "Goods vehicle 3.5–7.5t", 21: "Goods vehicle over 7.5t", 22: "Mobility scooter", 23: "Electric motorcycle", 90: "Other vehicle", 97: "Motor vehicle – unknown", 98: "Goods vehicle – unknown", 99: "Unknown", -1: "Data missing"}
USER_GROUP_LABELS = {"Pedestrian": "Pedestrians", "Cyclist": "Cyclists", "Motorcyclist": "Motorcyclists", "Car occupant": "Car occupants", "Other": "Other road users", "Unknown": "Unknown"}
USER_GROUP_ORDER = ["Pedestrian", "Cyclist", "Motorcyclist", "Car occupant", "Other", "Unknown"]
VULNERABLE_USER_GROUPS = ["Pedestrian", "Cyclist", "Motorcyclist"]
VULNERABLE = VULNERABLE_USER_GROUPS
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS = WEEKDAY_ORDER
HOUR_ORDER = [f"{h:02d}:00" for h in range(24)]
HOURS = HOUR_ORDER
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

MAP_SAMPLE_SIZE = 5000
EXPLORER_DEFAULT_ROWS = 25
MIN_GROUP_SIZE_FOR_INSIGHT = 100
RESEARCH_ONLY_NOTICE = "RESEARCH ONLY — This model has not passed its acceptance gate. Predictions are for research and demonstration only and must not be used for operational road-safety decisions."
