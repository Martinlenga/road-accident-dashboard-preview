"""Research-only collision severity prediction interface.

The saved model is for research and demonstration only. It must not be used
for operational road-safety decisions.
"""
import json
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from road_safety.config import MODELS_DIR, RED, AMBER, TEAL, BLUE, RESEARCH_ONLY_NOTICE
from road_safety.components import section_header, chart_card

MODEL_PATH = MODELS_DIR / "stats19_severity_model.joblib"
CONTRACT_PATH = MODELS_DIR / "model_contract.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
SEVERITY_NAMES = {1:"Fatal", 2:"Serious", 3:"Slight", "1":"Fatal", "2":"Serious", "3":"Slight"}
SEVERITY_COLOURS = {"Fatal":RED, "Serious":AMBER, "Slight":TEAL}
KEY_PREDICTORS = ["driver_age_mean", "vulnerable_road_user_count", "vulnerable_road_user_involved", "driver_age_min", "vehicle_type_first", "trunk_road_flag", "engine_cc_mean", "road_type", "time_band", "speed_limit"]
PREDICTOR_LABELS = {"driver_age_mean":"Mean driver age", "vulnerable_road_user_count":"Vulnerable road users involved", "vulnerable_road_user_involved":"Vulnerable road user involved", "driver_age_min":"Youngest driver age", "vehicle_type_first":"Primary vehicle type code", "trunk_road_flag":"Trunk road", "engine_cc_mean":"Mean engine capacity (cc)", "road_type":"Road type code", "time_band":"Time band code", "speed_limit":"Speed limit (mph)"}
DEFAULT_VALUES = {"driver_age_mean":38.0,"vulnerable_road_user_count":0,"vulnerable_road_user_involved":0,"driver_age_min":38.0,"vehicle_type_first":9,"trunk_road_flag":0,"engine_cc_mean":1600.0,"road_type":6,"time_band":2,"speed_limit":30,"motorcycle_involved":0,"goods_vehicle_involved":0,"weekend_flag":0}
CATEGORICAL_DEFAULTS = {"junction_detail":0,"junction_control":0,"light_conditions":1,"weather_conditions":1,"road_surface_conditions":1,"special_conditions_at_site":0,"carriageway_hazards":0,"urban_or_rural_area":1,"first_road_class":6,"second_road_class":6,"day_of_week":2,"month":6,"driver_sex_first":1,"vehicle_manoeuvre_first":18}
CODE_OPTIONS = {"road_type":[1,2,3,6,7,9,12],"vehicle_type_first":[1,2,3,4,5,8,9,10,11,19,20,21,22,23,97,98,103,104,105,106],"speed_limit":[20,30,40,50,60,70],"urban_or_rural_area":[1,2,3],"light_conditions":[1,4,5,6,7],"weather_conditions":[1,2,3,4,5,6,7,8,9],"road_surface_conditions":[1,2,3,4,5,6,7,9],"junction_detail":[0,1,2,3,5,6,7,8,9],"junction_control":[0,1,2,3,4],"first_road_class":[1,2,3,4,5,6],"second_road_class":[1,2,3,4,5,6],"day_of_week":[1,2,3,4,5,6,7],"month":list(range(1,13)),"driver_sex_first":[1,2,3],"vehicle_manoeuvre_first":list(range(1,19)),"time_band":[0,1,2,3,4,5]}

@st.cache_resource
def load_prediction_assets():
    for path in [MODEL_PATH, CONTRACT_PATH]:
        if not path.exists(): raise FileNotFoundError(f"Required prediction asset not found: {path}")
    model = joblib.load(MODEL_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    raw_features = contract.get("features")
    if not isinstance(raw_features, list) or not raw_features: raise ValueError("model_contract.json must contain a non-empty 'features' list.")
    names = []
    for item in raw_features:
        name = item if isinstance(item, str) else item.get("name") if isinstance(item, dict) else None
        if not name: raise ValueError(f"Invalid feature entry in model contract: {item!r}")
        names.append(name)
    if len(names) != len(set(names)): raise ValueError("Duplicate feature names found in model contract.")
    return model, contract, metadata, names


def _default_for(feature):
    return DEFAULT_VALUES.get(feature, CATEGORICAL_DEFAULTS.get(feature, 0))


def _label(feature):
    return PREDICTOR_LABELS.get(feature, feature.replace("_", " ").strip().title())


def _render_feature_input(feature, container, feature_spec=None):
    default = _default_for(feature)
    # Use contract-declared choices/ranges where the training contract provides them.
    spec = feature_spec if isinstance(feature_spec, dict) else {}
    choices = spec.get("categories") or spec.get("allowed_values") or CODE_OPTIONS.get(feature)
    if isinstance(choices, list) and choices:
        choices = [v for v in choices if v is not None]
        if not choices: choices = [default]
        index = choices.index(default) if default in choices else 0
        return container.selectbox(_label(feature), options=choices, index=index, key=f"prediction_{feature}", help=f"Value supplied to the model for {feature}.")
    if feature.endswith("_flag") or feature.endswith("_involved"):
        return container.selectbox(_label(feature), [0,1], index=int(default) if int(default) in [0,1] else 0, format_func=lambda x:"Yes" if x else "No", key=f"prediction_{feature}")
    if "age" in feature:
        return container.number_input(_label(feature), min_value=0.0, max_value=100.0, value=float(default), step=1.0, key=f"prediction_{feature}")
    if "engine" in feature or "capacity" in feature:
        return container.number_input(_label(feature), min_value=0.0, max_value=20000.0, value=float(default), step=100.0, key=f"prediction_{feature}")
    if "count" in feature:
        return container.number_input(_label(feature), min_value=0, max_value=100, value=int(default), step=1, key=f"prediction_{feature}")
    return container.number_input(_label(feature), value=float(default), step=1.0, key=f"prediction_{feature}", help="No specialised control is defined for this contract feature.")


def _model_classes(model):
    if hasattr(model, "classes_"): return list(model.classes_)
    if hasattr(model, "named_steps"):
        for estimator in reversed(list(model.named_steps.values())):
            if hasattr(estimator, "classes_"): return list(estimator.classes_)
    return [1,2,3]


def render_prediction_module():
    section_header("Severity Prediction", "Explore how the saved model classifies a hypothetical collision scenario.")
    st.warning(RESEARCH_ONLY_NOTICE, icon="⚠️")
    try:
        model, contract, metadata, feature_names = load_prediction_assets()
    except Exception as exc:
        st.error(f"Prediction assets could not be loaded: {exc}")
        st.info("Check that the saved estimator, model_contract.json and metadata.json exist in models/.")
        return
    if metadata.get("use_status"):
        st.caption(f"Model status: {metadata['use_status']}")
    st.markdown("Enter a hypothetical collision profile. The model receives the exact feature names and order specified by the training contract.")
    spec_by_name = {item.get("name"):item for item in contract.get("features", []) if isinstance(item, dict)}
    ordered = [f for f in KEY_PREDICTORS if f in feature_names] + [f for f in feature_names if f not in KEY_PREDICTORS]
    values = {}
    with st.form("severity_prediction_form"):
        st.markdown("#### Collision scenario")
        primary = [f for f in ordered if f in KEY_PREDICTORS]
        advanced = [f for f in ordered if f not in KEY_PREDICTORS]
        if primary:
            cols = st.columns(2)
            for i, feature in enumerate(primary): values[feature] = _render_feature_input(feature, cols[i%2], spec_by_name.get(feature))
        if advanced:
            with st.expander(f"Additional model inputs ({len(advanced)})"):
                cols = st.columns(2)
                for i, feature in enumerate(advanced): values[feature] = _render_feature_input(feature, cols[i%2], spec_by_name.get(feature))
        submitted = st.form_submit_button("Run severity prediction", type="primary", use_container_width=True)
    if not submitted:
        st.caption(f"The estimator expects {len(feature_names)} features. Inputs are assembled in the saved model contract order.")
        return
    input_frame = pd.DataFrame([[values.get(f, _default_for(f)) for f in feature_names]], columns=feature_names)
    try:
        prediction = model.predict(input_frame)[0]
        predicted_label = SEVERITY_NAMES.get(prediction, SEVERITY_NAMES.get(str(prediction), str(prediction)))
        st.markdown("---\n### Prediction result")
        colour = SEVERITY_COLOURS.get(predicted_label, BLUE)
        st.markdown(f"<div style='border-left:5px solid {colour};background:#F8FAFC;padding:20px 22px;border-radius:10px;margin:8px 0 18px'><div style='font-size:12px;color:#64748B;font-weight:700'>PREDICTED COLLISION SEVERITY</div><div style='font-size:30px;color:{colour};font-weight:800'>{html.escape(str(predicted_label))}</div></div>", unsafe_allow_html=True)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            classes = _model_classes(model)
            if len(classes) == len(probabilities):
                prob = pd.DataFrame({"Severity":[SEVERITY_NAMES.get(c, SEVERITY_NAMES.get(str(c), str(c))) for c in classes], "Probability (%)":np.asarray(probabilities)*100})
                chart_card("Model probability by severity class", "Model scores are not guarantees or necessarily calibrated real-world probabilities.")
                chart = alt.Chart(prob).mark_bar(cornerRadiusEnd=4).encode(x=alt.X("Probability (%):Q", title="Predicted probability (%)", scale=alt.Scale(domain=[0,100])), y=alt.Y("Severity:N", sort=["Fatal","Serious","Slight"], title=None), color=alt.Color("Severity:N", scale=alt.Scale(domain=["Fatal","Serious","Slight"], range=[RED,AMBER,TEAL]), legend=None), tooltip=["Severity", alt.Tooltip("Probability (%):Q", format=".2f")]).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
            else: st.info("Probability output could not be aligned to the estimator's class labels.")
        else: st.info("The saved estimator does not expose predict_proba(); probability scores are unavailable.")
        with st.expander("Review submitted model inputs"):
            st.dataframe(input_frame.T.rename(columns={0:"Input value"}), use_container_width=True)
    except Exception as exc:
        st.error(f"The model could not generate a prediction: {exc}")
        st.exception(exc)
