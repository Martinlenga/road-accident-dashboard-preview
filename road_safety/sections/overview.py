"""Executive overview of the filtered Stats19 records."""
import pandas as pd
import altair as alt
import streamlit as st
from road_safety.config import SEVERITY_ORDER, SEVERITY_COLOURS, AREA_COLOURS, MAP_SAMPLE_SIZE, BLUE, TEAL, AMBER, RED
from road_safety.components import section_header, kpi_row, chart_card, donut_chart, line_chart, bar_chart, insight_card
from road_safety.data import get_baseline_metrics
from road_safety.insights import monthly_trend, fatal_rate_by_speed, generate_insights, generate_recommendations


def render_overview(collisions, casualties, vehicles):
    section_header("Overview", "A high-level view of collision outcomes, road context and emerging patterns.")
    metrics = get_baseline_metrics(collisions, casualties)
    kpi_row([
        {"label":"Collisions", "value":f"{metrics['collisions']:,}", "context":"Recorded personal-injury collisions"},
        {"label":"Casualties", "value":f"{metrics['casualties']:,}", "context":"People recorded in linked casualty data"},
        {"label":"Fatal collisions", "value":f"{metrics['fatal_collisions']:,}", "context":f"{metrics['fatal_collisions']/max(metrics['collisions'],1)*100:.2f}% of collisions", "color":RED},
        {"label":"People killed", "value":f"{metrics['killed']:,}", "context":"Fatal casualty records", "color":AMBER},
        {"label":"Seriously injured", "value":f"{metrics['seriously_injured']:,}", "context":"Serious casualty records"},
    ])
    st.markdown("### Collision profile")
    left, right = st.columns(2)
    with left:
        severity = collisions.get("collision_severity_label", pd.Series(dtype="string")).value_counts().reindex(SEVERITY_ORDER, fill_value=0).rename_axis("Severity").reset_index(name="Collisions")
        chart_card("Collision severity", "Distribution of fatal, serious and slight collisions.")
        st.altair_chart(donut_chart(severity, "Severity", "Collisions", SEVERITY_COLOURS), use_container_width=True)
    with right:
        area = collisions.get("area_label", pd.Series(dtype="string")).value_counts().rename_axis("Area").reset_index(name="Collisions")
        chart_card("Urban and rural context")
        st.altair_chart(donut_chart(area, "Area", "Collisions", AREA_COLOURS), use_container_width=True)
    st.markdown("### Trends and risk context")
    left, right = st.columns(2)
    with left:
        trend = monthly_trend(collisions)
        chart_card("Monthly collision trend", "Monthly totals across the currently selected years.")
        if not trend.empty:
            st.altair_chart(line_chart(trend, "month:O", "collisions:Q"), use_container_width=True)
        else:
            st.info("No valid dates are available for the selected records.")
    with right:
        speed = fatal_rate_by_speed(collisions)
        chart_card("Fatal collision share by speed limit", "Only speed-limit groups meeting the minimum sample threshold are shown.")
        if not speed.empty:
            chart = alt.Chart(speed).mark_bar(cornerRadiusEnd=4, color=RED).encode(x=alt.X("speed_limit:O", title="Speed limit (mph)"), y=alt.Y("fatal_rate_pct:Q", title="Fatal collisions (%)"), tooltip=["speed_limit", "collisions", "fatal_collisions", alt.Tooltip("fatal_rate_pct:Q", format=".2f")]).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough valid speed-limit observations to calculate this comparison.")
    st.markdown("### Geographic distribution")
    if {"latitude", "longitude"}.issubset(collisions.columns):
        geo = collisions[["latitude", "longitude", "collision_severity_label"]].copy()
        geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
        geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
        geo = geo.dropna(subset=["latitude", "longitude"])
        geo = geo[geo["latitude"].between(49, 61) & geo["longitude"].between(-9, 3)]
        if len(geo) > MAP_SAMPLE_SIZE:
            geo = geo.sample(MAP_SAMPLE_SIZE, random_state=42)
        if not geo.empty:
            st.map(geo.rename(columns={"latitude":"lat", "longitude":"lon"})[["lat", "lon"]], size=8, color="#3977D5")
            st.caption(f"Showing {len(geo):,} valid UK coordinate points (sampled where necessary). This is a point distribution, not an exposure-adjusted hotspot analysis.")
        else:
            st.info("No valid UK coordinates are available in the current selection.")
    findings = generate_insights(collisions, casualties)
    if findings:
        st.markdown("### Key observations")
        for title, message in findings:
            insight_card(title, message)
    st.markdown("### Interpretation")
    for recommendation in generate_recommendations(collisions, casualties):
        st.markdown(f"- {recommendation}")
    st.caption("These are descriptive patterns in recorded collisions. They do not establish causation or account for traffic exposure, reporting differences or changes in travel volume.")


def render(collisions, casualties, vehicles):
    return render_overview(collisions, casualties, vehicles)
