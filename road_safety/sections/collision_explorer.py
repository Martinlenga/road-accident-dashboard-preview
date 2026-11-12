"""Filterable collision register with linked casualty and vehicle detail."""
import pandas as pd
import streamlit as st
from road_safety.config import EXPLORER_DEFAULT_ROWS
from road_safety.components import section_header

DISPLAY_COLUMNS = [
    "collision_index", "date", "time", "collision_severity_label", "area_label",
    "road_type_label", "speed_limit", "light_label", "weather_conditions_label",
    "road_surface_conditions_label", "number_of_vehicles", "number_of_casualties",
]


def render_collision_explorer(collisions, casualties, vehicles):
    section_header("Collision Explorer", "Search the collision register and inspect the linked people and vehicle records.")
    if collisions.empty:
        st.info("No collision records match the current dashboard filters.")
        return
    frame = collisions.copy()
    with st.expander("Explorer filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        severity_values = sorted(frame.get("collision_severity_label", pd.Series(dtype=str)).dropna().unique().tolist())
        area_values = sorted(frame.get("area_label", pd.Series(dtype=str)).dropna().unique().tolist())
        road_values = sorted(frame.get("road_type_label", pd.Series(dtype=str)).dropna().unique().tolist())
        selected_severity = c1.multiselect("Severity", severity_values, default=severity_values, key="explorer_severity")
        selected_area = c2.multiselect("Area", area_values, default=area_values, key="explorer_area")
        selected_road = c3.multiselect("Road type", road_values, default=road_values, key="explorer_road")
        search = st.text_input("Search collision ID or location text", key="explorer_search").strip().lower()
    if selected_severity and "collision_severity_label" in frame:
        frame = frame[frame["collision_severity_label"].isin(selected_severity)]
    if selected_area and "area_label" in frame:
        frame = frame[frame["area_label"].isin(selected_area)]
    if selected_road and "road_type_label" in frame:
        frame = frame[frame["road_type_label"].isin(selected_road)]
    if search:
        searchable = [c for c in ["collision_index", "local_authority_district", "local_authority_ons_district", "local_authority_highway", "lsoa_of_accident_location"] if c in frame]
        if searchable:
            mask = pd.Series(False, index=frame.index)
            for col in searchable:
                mask |= frame[col].astype("string").str.lower().str.contains(search, na=False)
            frame = frame[mask]
    st.caption(f"{len(frame):,} matching collisions")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="explorer_page_size")
    max_page = max(1, (len(frame) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="explorer_page")
    start = (int(page) - 1) * page_size
    visible_cols = [c for c in DISPLAY_COLUMNS if c in frame.columns]
    st.dataframe(frame.iloc[start:start+page_size][visible_cols], use_container_width=True, hide_index=True)
    if "collision_index" not in frame:
        return
    st.markdown("### Collision record detail")
    ids = frame["collision_index"].dropna().astype(str).tolist()
    if not ids:
        st.info("No collision identifiers available for drill-down.")
        return
    selected_id = st.selectbox("Select a collision", ids, format_func=lambda x: f"Collision {x}", key="explorer_selected_collision")
    selected = frame[frame["collision_index"].astype(str).eq(str(selected_id))].head(1)
    if not selected.empty:
        record = selected.iloc[0]
        detail_cols = st.columns(3)
        detail_fields = [("Severity", "collision_severity_label"), ("Date", "date"), ("Time", "time"), ("Area", "area_label"), ("Road type", "road_type_label"), ("Speed limit", "speed_limit"), ("Lighting", "light_label"), ("Weather", "weather_conditions_label"), ("Casualties", "number_of_casualties")]
        for i, (label, col) in enumerate(detail_fields):
            if col in selected.columns:
                with detail_cols[i % 3]: st.metric(label, str(record[col]) if pd.notna(record[col]) else "Unknown")
    casualty_detail = casualties[casualties["collision_index"].astype(str).eq(str(selected_id))].copy() if "collision_index" in casualties else pd.DataFrame()
    vehicle_detail = vehicles[vehicles["collision_index"].astype(str).eq(str(selected_id))].copy() if "collision_index" in vehicles else pd.DataFrame()
    tab1, tab2 = st.tabs([f"Casualties ({len(casualty_detail):,})", f"Vehicles ({len(vehicle_detail):,})"])
    with tab1:
        cols = [c for c in ["casualty_severity_label", "casualty_class_label", "age_band_label", "sex_label", "casualty_type_label", "user_type"] if c in casualty_detail]
        st.dataframe(casualty_detail[cols] if cols else casualty_detail, use_container_width=True, hide_index=True)
    with tab2:
        cols = [c for c in ["vehicle_type_label", "driver_sex_label", "driver_age_band_label", "vehicle_manoeuvre_label", "engine_capacity_cc"] if c in vehicle_detail]
        st.dataframe(vehicle_detail[cols] if cols else vehicle_detail, use_container_width=True, hide_index=True)


def render(collisions, casualties, vehicles):
    return render_collision_explorer(collisions, casualties, vehicles)
