"""Temporal and roadway-condition analysis."""
import pandas as pd
import altair as alt
import streamlit as st
from road_safety.config import WEEKDAY_ORDER, MONTH_ORDER, SEVERITY_ORDER, SEVERITY_COLOURS, BLUE, TEAL
from road_safety.components import section_header, kpi_row, chart_card, heatmap, bar_chart, stacked_bar_chart


def _count_chart(frame, category, title, limit=15):
    if category not in frame:
        st.info(f"{title}: source field is unavailable.")
        return
    counts = frame[category].fillna("Unknown").astype(str).value_counts().head(limit).rename_axis("Category").reset_index(name="Collisions")
    chart_card(title)
    if counts.empty:
        st.info("No records to display.")
    else:
        chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, color=BLUE).encode(x=alt.X("Collisions:Q", title="Collisions"), y=alt.Y("Category:N", sort="-x", title=None), tooltip=["Category", "Collisions"]).properties(height=max(180, min(360, len(counts)*25)))
        st.altair_chart(chart, use_container_width=True)


def render_collision_analytics(collisions):
    section_header("Collision Analytics", "Explore when collisions occur and how road, lighting and weather conditions vary.")
    total = len(collisions)
    weekend = int(collisions.get("is_weekend", pd.Series(False, index=collisions.index)).sum())
    dark = int(collisions.get("is_dark", pd.Series(False, index=collisions.index)).sum())
    kpi_row([
        {"label":"Selected collisions", "value":f"{total:,}"},
        {"label":"Weekend collisions", "value":f"{weekend:,}", "context":f"{weekend/max(total,1)*100:.1f}% of selection"},
        {"label":"Darkness conditions", "value":f"{dark:,}", "context":f"{dark/max(total,1)*100:.1f}% of selection"},
    ])
    st.markdown("### Time patterns")
    left, right = st.columns(2)
    with left:
        if {"hour", "weekday"}.issubset(collisions.columns):
            hourly = collisions.dropna(subset=["hour"]).groupby(["weekday", "hour"]).size().rename("Collisions").reset_index()
            hourly["hour_label"] = hourly["hour"].astype(int).map(lambda h:f"{h:02d}:00")
            chart_card("Day and hour distribution", "Darker cells represent higher collision counts.")
            if not hourly.empty:
                st.altair_chart(heatmap(hourly, "hour_label", "weekday", "Collisions"), use_container_width=True)
            else: st.info("No valid time values in this selection.")
    with right:
        if "month_label" in collisions:
            monthly = collisions[collisions["month_label"].isin(MONTH_ORDER)].groupby("month_label").size().reindex(MONTH_ORDER, fill_value=0).rename("Collisions").reset_index().rename(columns={"index":"Month", "month_label":"Month"})
            chart_card("Seasonal distribution", "Counts aggregated by calendar month across selected years.")
            chart = alt.Chart(monthly).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(x=alt.X("Month:N", sort=MONTH_ORDER, title=None, axis=alt.Axis(labelAngle=-35)), y=alt.Y("Collisions:Q", title="Collisions"), tooltip=["Month", "Collisions"]).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
    st.markdown("### Road environment")
    a, b = st.columns(2)
    with a: _count_chart(collisions, "road_type_label", "Collision count by road type")
    with b: _count_chart(collisions, "speed_limit", "Collision count by speed limit")
    st.markdown("### Conditions and severity")
    for label, column in [("Lighting", "light_label"), ("Weather", "weather_conditions_label"), ("Road surface", "road_surface_conditions_label"), ("Junction detail", "junction_detail_label"), ("Junction control", "junction_control_label")]:
        if column in collisions:
            left, right = st.columns([1, 1.5])
            with left: _count_chart(collisions, column, f"{label} conditions", limit=10)
            with right:
                grouped = collisions.groupby([column, "collision_severity_label"]).size().rename("Collisions").reset_index().rename(columns={column:"Condition"})
                chart_card(f"Severity by {label.lower()}", "100% stacked bars compare severity composition within each category.")
                chart = alt.Chart(grouped).mark_bar().encode(x=alt.X("Condition:N", title=None, axis=alt.Axis(labelAngle=-35)), y=alt.Y("Collisions:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of collisions"), color=alt.Color("collision_severity_label:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=[SEVERITY_COLOURS[s] for s in SEVERITY_ORDER]), legend=alt.Legend(title="Severity")), tooltip=["Condition", alt.Tooltip("collision_severity_label:N", title="Severity"), "Collisions"]).properties(height=280)
                st.altair_chart(chart, use_container_width=True)
    st.caption("Collision counts are not rates. Comparisons do not adjust for traffic volume, road length, journey exposure or reporting differences.")


def render(collisions):
    return render_collision_analytics(collisions)
