"""Small, transparent descriptive-analysis helpers."""
import pandas as pd
from road_safety.config import MIN_GROUP_SIZE_FOR_INSIGHT, VULNERABLE_USER_GROUPS


def safe_percentage(numerator, denominator):
    return float(numerator / denominator * 100) if denominator else 0.0


def fatal_rate_by_speed(collisions, minimum_group_size=MIN_GROUP_SIZE_FOR_INSIGHT):
    if not {"speed_limit", "collision_severity_label"}.issubset(collisions.columns):
        return pd.DataFrame(columns=["speed_limit", "collisions", "fatal_collisions", "fatal_rate_pct"])
    frame = collisions.copy()
    frame["speed_limit"] = pd.to_numeric(frame["speed_limit"], errors="coerce")
    frame = frame[frame["speed_limit"].gt(0)]
    if frame.empty:
        return pd.DataFrame(columns=["speed_limit", "collisions", "fatal_collisions", "fatal_rate_pct"])
    result = frame.groupby("speed_limit").agg(collisions=("collision_severity_label", "size"), fatal_collisions=("collision_severity_label", lambda s: s.eq("Fatal").sum())).reset_index()
    result = result[result["collisions"] >= minimum_group_size]
    result["fatal_rate_pct"] = result["fatal_collisions"] / result["collisions"] * 100
    return result.sort_values("speed_limit")


def road_user_ksi(casualties, minimum_group_size=MIN_GROUP_SIZE_FOR_INSIGHT):
    if not {"user_type", "is_ksi"}.issubset(casualties.columns):
        return pd.DataFrame(columns=["user_type", "casualties", "ksi", "ksi_share_pct"])
    result = casualties.groupby("user_type").agg(casualties=("is_ksi", "size"), ksi=("is_ksi", "sum")).reset_index()
    result = result[result["casualties"] >= minimum_group_size].copy()
    result["ksi_share_pct"] = result["ksi"] / result["casualties"] * 100
    return result.sort_values("ksi_share_pct", ascending=False)


def monthly_trend(collisions):
    if "date" not in collisions:
        return pd.DataFrame(columns=["month", "collisions"])
    dates = pd.to_datetime(collisions["date"], errors="coerce")
    frame = pd.DataFrame({"month": dates.dt.to_period("M").astype("string")}).dropna()
    return frame.groupby("month").size().rename("collisions").reset_index().sort_values("month")


def time_distribution(collisions):
    if "hour" not in collisions:
        return pd.DataFrame(columns=["hour", "weekday", "collisions"])
    return collisions.dropna(subset=["hour", "weekday"]).groupby(["hour", "weekday"]).size().rename("collisions").reset_index()


def generate_insights(collisions, casualties):
    findings = []
    n = len(collisions)
    if n and "collision_severity_label" in collisions:
        fatal = int(collisions["collision_severity_label"].eq("Fatal").sum())
        findings.append(("Fatal collision share", f"{safe_percentage(fatal, n):.2f}% of the selected collisions were recorded as fatal."))
    if len(casualties) and "is_vru" in casualties:
        share = safe_percentage(int(casualties["is_vru"].sum()), len(casualties))
        findings.append(("Vulnerable road users", f"Pedestrians, cyclists and motorcyclists account for {share:.1f}% of casualties in this selection."))
    if n and "is_dark" in collisions:
        findings.append(("Lighting context", f"{safe_percentage(int(collisions['is_dark'].sum()), n):.1f}% of collisions occurred in recorded darkness conditions."))
    speed = fatal_rate_by_speed(collisions)
    if not speed.empty:
        row = speed.loc[speed["fatal_rate_pct"].idxmax()]
        findings.append(("Speed-limit pattern", f"The {int(row['speed_limit'])} mph group has the highest observed fatal share ({row['fatal_rate_pct']:.2f}%) among speed groups meeting the minimum sample threshold."))
    return findings


def generate_recommendations(collisions, casualties):
    recs = []
    if len(casualties) and "is_vru" in casualties and casualties["is_vru"].mean() > 0.2:
        recs.append("Prioritise pedestrian, cycling and motorcycle safety measures in locations with high vulnerable-road-user exposure.")
    if len(collisions) and "is_dark" in collisions and collisions["is_dark"].mean() > 0.2:
        recs.append("Review lighting, visibility and night-time road-user protection in areas with elevated darkness-related collision counts.")
    if not recs:
        recs.append("Use the filtered patterns to identify local priorities, then validate them with exposure, traffic-volume and road-engineering evidence.")
    return recs
