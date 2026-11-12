"""Casualty, driver and vehicle analysis."""
import pandas as pd
import altair as alt
import streamlit as st
from road_safety.config import USER_GROUP_ORDER, AGE_BAND_ORDER, SEVERITY_ORDER, SEVERITY_COLOURS, USER_GROUP_LABELS, VULNERABLE_USER_GROUPS, BLUE, TEAL, AMBER
from road_safety.components import section_header, kpi_row, chart_card


def _bar(frame, category, value, title, order=None, color=BLUE, height=280):
    chart_card(title)
    if frame.empty or category not in frame or value not in frame:
        st.info("No data available for this view.")
        return
    sort = order if order else "-x"
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3, color=color).encode(x=alt.X(f"{value}:Q", title="People" if value != "KSI share (%)" else value), y=alt.Y(f"{category}:N", sort=sort, title=None), tooltip=list(frame.columns)).properties(height=height)
    st.altair_chart(chart, use_container_width=True)


def render_road_users(collisions, casualties, vehicles):
    section_header("Road Users", "Understand who is involved, who is injured and how outcomes differ across road-user groups.")
    total = len(casualties)
    ksi = int(casualties.get("is_ksi", pd.Series(False, index=casualties.index)).sum())
    vru = int(casualties.get("is_vru", pd.Series(False, index=casualties.index)).sum())
    kpi_row([
        ("Casualties", f"{total:,}", "Linked casualty records"),
        ("Killed or seriously injured", f"{ksi:,}", f"{ksi/max(total,1)*100:.1f}% of casualties"),
        ("Vulnerable road users", f"{vru:,}", f"{vru/max(total,1)*100:.1f}% of casualties"),
        ("Vehicle records", f"{len(vehicles):,}", "Linked vehicle records"),
    ])
    st.markdown("### Road-user groups")
    left, right = st.columns(2)
    with left:
        group = casualties.get("user_type", pd.Series(dtype="string")).value_counts().reindex(USER_GROUP_ORDER, fill_value=0).rename_axis("Road user").reset_index(name="Casualties")
        _bar(group, "Road user", "Casualties", "Casualties by road-user group", order=USER_GROUP_ORDER)
    with right:
        if {"user_type", "is_ksi"}.issubset(casualties.columns):
            ksi_group = casualties.groupby("user_type").agg(Casualties=("is_ksi", "size"), KSI=("is_ksi", "sum")).reset_index()
            ksi_group["KSI share (%)"] = ksi_group["KSI"] / ksi_group["Casualties"] * 100
            chart_card("KSI share within each road-user group", "KSI means killed or seriously injured casualties.")
            chart = alt.Chart(ksi_group).mark_bar(color=AMBER, cornerRadiusEnd=3).encode(x=alt.X("KSI share (%):Q", title="KSI share (%)", scale=alt.Scale(domain=[0,100])), y=alt.Y("user_type:N", sort="-x", title=None), tooltip=["user_type", "Casualties", "KSI", alt.Tooltip("KSI share (%):Q", format=".1f")]).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
    st.markdown("### Demographics")
    left, right = st.columns(2)
    with left:
        if "age_band_label" in casualties:
            age = casualties["age_band_label"].value_counts().reindex(AGE_BAND_ORDER, fill_value=0).rename_axis("Age band").reset_index(name="Casualties")
            _bar(age, "Age band", "Casualties", "Casualties by age band", order=AGE_BAND_ORDER)
    with right:
        if "sex_label" in casualties:
            sex = casualties["sex_label"].value_counts().rename_axis("Sex").reset_index(name="Casualties")
            _bar(sex, "Sex", "Casualties", "Casualties by recorded sex")
    st.markdown("### Severity and vehicle profile")
    left, right = st.columns(2)
    with left:
        if "casualty_severity_label" in casualties:
            sev = casualties["casualty_severity_label"].value_counts().reindex(SEVERITY_ORDER, fill_value=0).rename_axis("Severity").reset_index(name="Casualties")
            chart_card("Casualty severity")
            chart = alt.Chart(sev).mark_bar(cornerRadiusEnd=3).encode(x=alt.X("Severity:N", sort=SEVERITY_ORDER, title=None), y=alt.Y("Casualties:Q", title="People"), color=alt.Color("Severity:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=[SEVERITY_COLOURS[s] for s in SEVERITY_ORDER]), legend=None), tooltip=["Severity", "Casualties"]).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
    with right:
        if "vehicle_type_label" in vehicles:
            vtypes = vehicles["vehicle_type_label"].value_counts().head(12).rename_axis("Vehicle type").reset_index(name="Vehicles")
            _bar(vtypes, "Vehicle type", "Vehicles", "Most common vehicle types", height=320)
    if "user_type" in casualties:
        st.markdown("### Vulnerable road users")
        vru_frame = casualties[casualties["user_type"].isin(VULNERABLE_USER_GROUPS)]
        if vru_frame.empty:
            st.info("No vulnerable-road-user casualties in this selection.")
        else:
            vru_summary = vru_frame.groupby("user_type").agg(Casualties=("is_ksi", "size"), KSI=("is_ksi", "sum")).reset_index()
            vru_summary["KSI share (%)"] = vru_summary["KSI"] / vru_summary["Casualties"] * 100
            st.dataframe(vru_summary, hide_index=True, use_container_width=True)
    st.caption("Casualty and vehicle records are linked to collision records by collision_index. A person-level casualty count is not equivalent to a collision count.")


def render(collisions, casualties, vehicles):
    return render_road_users(collisions, casualties, vehicles)
