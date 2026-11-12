"""Reusable Streamlit and Altair presentation components."""
import html
import altair as alt
import pandas as pd
import streamlit as st
from road_safety.config import INK, BLUE, TEAL, AMBER, RED, GREY, WHITE, SEVERITY_ORDER, SEVERITY_COLOURS


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"<div style='margin:4px 0 18px'><div style='font-size:25px;font-weight:800;color:{INK};'>{html.escape(title)}</div><div style='color:#64748B;font-size:13px;margin-top:4px'>{html.escape(subtitle)}</div></div>", unsafe_allow_html=True)


def kpi_card(label, value, context="", color=BLUE, help_text=None):
    st.metric(label=label, value=value, help=help_text)
    if context:
        st.caption(context)


def kpi_row(items):
    """Render KPI dictionaries or legacy (label, value, context) tuples."""
    normalised = []
    for item in items:
        if isinstance(item, dict):
            normalised.append(item)
        else:
            values = list(item)
            normalised.append({"label": values[0], "value": values[1], "context": values[2] if len(values) > 2 else ""})
    cols = st.columns(max(1, len(normalised)))
    for col, item in zip(cols, normalised):
        with col:
            kpi_card(item.get("label", "Metric"), item.get("value", "—"), item.get("context", ""), item.get("color", BLUE), item.get("help"))


def chart_card(title, subtitle=""):
    """Display a chart heading; chart objects are rendered directly by callers."""
    if isinstance(title, alt.Chart):
        st.altair_chart(title, use_container_width=True)
        return
    st.markdown(f"<div style='font-weight:700;color:{INK};margin:10px 0 2px'>{html.escape(str(title))}</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def chart_theme(chart, height=300):
    return chart.properties(height=height).configure_view(strokeOpacity=0).configure_axis(labelColor="#64748B", titleColor="#475569", gridColor="#E8EDF4", domainColor="#CBD5E1", labelFontSize=11, titleFontSize=11).configure_title(color=INK, fontSize=14, anchor="start").configure_legend(labelColor="#475569", titleColor="#475569")


def bar_chart(data, x, y, color=None, title=None, height=300, sort="-y"):
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(x, title=None), y=alt.Y(y, title=None, sort=sort),
        color=alt.Color(color, legend=alt.Legend(title=None)) if color else alt.value(BLUE),
        tooltip=[alt.Tooltip(field, type="quantitative" if pd.api.types.is_numeric_dtype(data[field]) else "nominal") for field in data.columns]
    )
    return chart_theme(chart, height).properties(title=title) if title else chart_theme(chart, height)


def line_chart(data, x, y, color=None, title=None, height=300):
    chart = alt.Chart(data).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X(x, title=None), y=alt.Y(y, title=None),
        color=alt.Color(color, legend=alt.Legend(title=None)) if color else alt.value(TEAL),
        tooltip=list(data.columns)
    )
    return chart_theme(chart, height).properties(title=title) if title else chart_theme(chart, height)


def donut_chart(data, category, value, colors=None, title=None, height=280):
    color = alt.Color(f"{category}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=alt.Legend(title=None)) if colors else alt.Color(f"{category}:N", legend=alt.Legend(title=None))
    chart = alt.Chart(data).mark_arc(innerRadius=70, outerRadius=112).encode(
        theta=alt.Theta(f"{value}:Q", stack=True), color=color,
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=",")]
    )
    return chart_theme(chart, height).properties(title=title) if title else chart_theme(chart, height)


def stacked_bar_chart(data, x, y, color, normalize=False, title=None, height=320):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x}:N", title=None),
        y=alt.Y(f"{y}:Q", stack="normalize" if normalize else "zero", title="Share of records" if normalize else "Number of collisions", axis=alt.Axis(format="%" if normalize else "~s")),
        color=alt.Color(f"{color}:N", scale=alt.Scale(domain=SEVERITY_ORDER, range=[SEVERITY_COLOURS[s] for s in SEVERITY_ORDER]), legend=alt.Legend(title="Severity")),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{color}:N"), alt.Tooltip(f"{y}:Q", format=",")]
    )
    return chart_theme(chart, height).properties(title=title) if title else chart_theme(chart, height)


def heatmap(data, x, y, value, title=None, height=340):
    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X(f"{x}:O", title=None), y=alt.Y(f"{y}:O", title=None),
        color=alt.Color(f"{value}:Q", scale=alt.Scale(scheme="blues"), title="Collisions"),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{value}:Q", format=",")]
    )
    return chart_theme(chart, height).properties(title=title) if title else chart_theme(chart, height)


def insight_card(title, message, kind="info"):
    palette = {"info": BLUE, "success": TEAL, "warning": AMBER, "danger": RED}
    accent = palette.get(kind, BLUE)
    st.markdown(f"<div style='border-left:4px solid {accent};background:#FFFFFF;padding:12px 15px;border-radius:8px;margin:8px 0'><div style='font-weight:700;color:{INK};margin-bottom:4px'>{html.escape(str(title))}</div><div style='font-size:13px;color:#475569'>{html.escape(str(message))}</div></div>", unsafe_allow_html=True)
