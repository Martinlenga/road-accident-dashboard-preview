"""Transparent reporting of saved model evaluation artefacts."""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from road_safety.config import MODELS_DIR, OUTPUTS_DIR, RESEARCH_ONLY_NOTICE, SEVERITY_ORDER, SEVERITY_COLOURS, INK, BLUE
from road_safety.components import section_header, kpi_row, chart_card


def _read_json(path):
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError): return {}
    return {}


def _walk_dicts(obj, path=""):
    if isinstance(obj, dict):
        yield path, obj
        for key, value in obj.items(): yield from _walk_dicts(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj): yield from _walk_dicts(value, f"{path}[{i}]")


def _metric_map(metadata):
    """Extract only explicitly named evaluation metrics; ignore arbitrary numeric metadata."""
    aliases = {
        "accuracy": ["accuracy", "test_accuracy"],
        "balanced_accuracy": ["balanced_accuracy", "test_balanced_accuracy"],
        "macro_f1": ["macro_f1", "f1_macro", "test_macro_f1"],
        "weighted_f1": ["weighted_f1", "f1_weighted", "test_weighted_f1"],
        "fatal_precision": ["fatal_precision"], "fatal_recall": ["fatal_recall"],
        "serious_precision": ["serious_precision"], "serious_recall": ["serious_recall"],
        "slight_precision": ["slight_precision"], "slight_recall": ["slight_recall"],
    }
    found = {}
    for _, node in _walk_dicts(metadata):
        for canonical, names in aliases.items():
            for name in names:
                value = node.get(name)
                if isinstance(value, (int,float)) and np.isfinite(value) and 0 <= value <= 1:
                    found[canonical] = float(value)
    return found


def _find_file(names):
    for folder in [MODELS_DIR, OUTPUTS_DIR]:
        for name in names:
            path = folder / name
            if path.exists(): return path
    return None


def _load_confusion_matrix():
    path = _find_file(["confusion_matrix.json", "confusion_matrix.csv", "test_confusion_matrix.csv"])
    if path is None: return None
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            matrix = raw.get("matrix", raw.get("confusion_matrix", raw)) if isinstance(raw,dict) else raw
            labels = raw.get("labels", SEVERITY_ORDER) if isinstance(raw,dict) else SEVERITY_ORDER
            frame = pd.DataFrame(matrix, index=labels, columns=labels)
        else:
            frame = pd.read_csv(path, index_col=0)
        return frame
    except Exception:
        return None


def _feature_importance():
    model_path = MODELS_DIR / "stats19_severity_model.joblib"
    contract = _read_json(MODELS_DIR / "model_contract.json")
    feature_names = [x if isinstance(x,str) else x.get("name") for x in contract.get("features",[]) if isinstance(x,(str,dict))]
    if not model_path.exists(): return None
    try:
        model = joblib.load(model_path)
        estimator = model
        if hasattr(model, "named_steps"):
            for candidate in reversed(list(model.named_steps.values())):
                if hasattr(candidate, "feature_importances_") or hasattr(candidate, "coef_"):
                    estimator = candidate; break
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_).ravel()
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_)).mean(axis=0).ravel()
        else: return None
        if len(values) != len(feature_names): return None
        return pd.DataFrame({"Feature":feature_names, "Importance":values}).sort_values("Importance", ascending=False).head(20)
    except Exception:
        return None


def render_model_insights():
    section_header("Model Insights", "Evaluation evidence, model behaviour and limitations of the saved severity classifier.")
    st.warning(RESEARCH_ONLY_NOTICE, icon="⚠️")
    metadata = _read_json(MODELS_DIR / "metadata.json")
    status = metadata.get("use_status") or metadata.get("acceptance_decision")
    if status: st.info(f"Recorded model status: {status}")
    metrics = _metric_map(metadata)
    if metrics:
        display = [("Macro F1", "macro_f1"), ("Balanced accuracy", "balanced_accuracy"), ("Accuracy", "accuracy"), ("Weighted F1", "weighted_f1")]
        items = [{"label":label, "value":f"{metrics[key]:.3f}"} for label,key in display if key in metrics]
        if items: kpi_row(items)
        st.markdown("### Class-level performance")
        rows = []
        for severity in ["fatal", "serious", "slight"]:
            row = {"Severity":severity.title()}
            for metric in ["precision", "recall"]:
                key=f"{severity}_{metric}"
                if key in metrics: row[metric.title()] = metrics[key]
            if len(row)>1: rows.append(row)
        if rows:
            perf = pd.DataFrame(rows)
            st.dataframe(perf.style.format({c:"{:.3f}" for c in perf.columns if c != "Severity"}), hide_index=True, use_container_width=True)
    else:
        st.info("No recognised evaluation metrics were found in metadata.json. No performance values have been inferred or substituted.")
    st.markdown("### Confusion matrix")
    matrix = _load_confusion_matrix()
    if matrix is not None and not matrix.empty:
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows and columns retain the ordering recorded in the saved confusion-matrix artefact.")
    else:
        st.info("No readable confusion-matrix artefact was found in models/ or outputs/. Add the matrix exported during evaluation to display it here.")
    st.markdown("### Feature importance")
    importance = _feature_importance()
    if importance is not None and not importance.empty:
        chart = alt.Chart(importance).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(x=alt.X("Importance:Q", title="Importance / mean absolute coefficient"), y=alt.Y("Feature:N", sort="-x", title=None), tooltip=["Feature", alt.Tooltip("Importance:Q", format=".5f")]).properties(height=max(300,len(importance)*22))
        st.altair_chart(chart, use_container_width=True)
        st.caption("For tree estimators this is the estimator's feature_importances_. For linear estimators it is mean absolute coefficient magnitude. These measures are not causal effects.")
    else:
        st.info("Feature importance could not be derived safely from the saved estimator and model contract.")
    st.markdown("### Model scope and limitations")
    limitations = [
        "The model estimates severity conditional on a recorded personal-injury collision; it does not predict whether or where a collision will occur.",
        "The evaluation uses a chronological hold-out period. Performance may change under different years, regions, road networks or reporting practices.",
        "Fatal collisions are a small minority class. Aggregate accuracy can conceal weak performance on rare but important outcomes.",
        "Predictions depend on the feature definitions and input quality used during training. Hypothetical inputs may not represent realistic combinations.",
        "The model has not passed its acceptance gate and is not approved for operational decisions, enforcement, resource allocation or public-safety interventions.",
    ]
    for item in limitations: st.markdown(f"- {item}")
    st.caption("Metrics shown here are read from the saved metadata and artefacts. Missing evaluation artefacts are deliberately left undisplayed rather than fabricated.")


def render():
    return render_model_insights()
