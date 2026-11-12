"""Shared sidebar filters and linked-table filtering."""
import pandas as pd
import streamlit as st
from road_safety.config import SEVERITY_ORDER


def render_global_filters(collisions):
    st.sidebar.markdown("## Dashboard filters")
    st.sidebar.caption("Selections apply across all analytical sections.")
    filtered = collisions.copy()
    years = sorted(filtered["date"].dropna().dt.year.unique().tolist()) if "date" in filtered else []
    selected_years = st.sidebar.multiselect("Collision year", years, default=years, key="global_years")
    if "date" in filtered and selected_years:
        filtered = filtered[filtered["date"].dt.year.isin(selected_years)]
    severities = [v for v in SEVERITY_ORDER if v in filtered.get("collision_severity_label", pd.Series(dtype=str)).astype(str).unique()]
    selected_severity = st.sidebar.multiselect("Collision severity", severities, default=severities, key="global_severity")
    if selected_severity and "collision_severity_label" in filtered:
        filtered = filtered[filtered["collision_severity_label"].astype(str).isin(selected_severity)]
    for label, column, key in [("Area type","area_label","global_areas"),("Road type","road_type_label","global_roads"),("Speed limit (mph)","speed_label","global_speeds")]:
        options = sorted(filtered[column].dropna().astype(str).unique().tolist()) if column in filtered else []
        if column == "speed_label": options = sorted(options, key=lambda x:(999 if x=="Unknown" else int(x) if x.isdigit() else 998))
        selected = st.sidebar.multiselect(label, options, default=options, key=key)
        if selected and column in filtered: filtered = filtered[filtered[column].astype(str).isin(selected)]
    st.sidebar.divider()
    st.sidebar.metric("Collisions in current selection", f"{len(filtered):,}")
    if st.sidebar.button("Reset all filters", use_container_width=True):
        for key in ["global_years","global_severity","global_areas","global_roads","global_speeds"]: st.session_state.pop(key, None)
        st.rerun()
    return filtered


def filter_related_data(collisions, casualties, vehicles):
    ids = set(collisions["collision_index"].dropna().astype(str)) if "collision_index" in collisions else set()
    casualty_view = casualties[casualties["collision_index"].astype(str).isin(ids)].copy() if "collision_index" in casualties else casualties.iloc[0:0].copy()
    vehicle_view = vehicles[vehicles["collision_index"].astype(str).isin(ids)].copy() if "collision_index" in vehicles else vehicles.iloc[0:0].copy()
    return casualty_view, vehicle_view


def render_filter_summary(collisions):
    if collisions.empty:
        st.warning("No collision records match the current filter selection.")
        return
    dates = collisions["date"].dropna() if "date" in collisions else pd.Series(dtype="datetime64[ns]")
    date_range = f"{dates.min():%d %b %Y} – {dates.max():%d %b %Y}" if not dates.empty else "date range unavailable"
    st.caption(f"Currently analysing **{len(collisions):,} collisions** ({date_range}).")
