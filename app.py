import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import load_csv, preprocess, validate_dataset
from utils.ml_utils import (
    DEFAULT_TARGET,
    PREDICTION_TARGETS,
    best_model_name,
    default_feature_columns,
    feature_importance,
    predict,
    train_and_evaluate,
)

st.set_page_config(
    page_title="5G Network Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: linear-gradient(135deg, rgba(38,45,66,.95), rgba(22,27,42,.95)); border: 1px solid rgba(120,140,180,.18); padding: 16px 18px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,.12);}
    [data-testid="stMetricLabel"] {font-size: .82rem; opacity: .75;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .hero {padding: 22px 26px; border-radius: 22px; background: radial-gradient(circle at 85% 20%, rgba(0,198,255,.22), transparent 30%), linear-gradient(135deg,#101827,#19243a); border: 1px solid rgba(95,170,255,.18); margin-bottom: 18px;}
    .hero h1 {margin: 0 0 6px 0; font-size: 2.25rem;}
    .hero p {margin: 0; opacity: .78; font-size: 1rem;}
    .section-title {font-size: 1.15rem; font-weight: 700; margin: 22px 0 10px 0;}
    .pill {display:inline-block; padding:5px 10px; border-radius:999px; background:rgba(0,198,255,.12); border:1px solid rgba(0,198,255,.22); font-size:.78rem; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PATH = ROOT / "data" / "5g_network_data.csv"


def load_data():
    uploaded = st.session_state.get("uploaded_csv")
    if uploaded is not None:
        return load_csv(uploaded)
    if DEFAULT_PATH.exists():
        return load_csv(DEFAULT_PATH)
    return None


def quality_score(df):
    # A transparent 0-100 operational score for dashboard comparison.
    def norm_high(x, best):
        return np.clip(x / best * 100, 0, 100)

    def norm_low(x, best, worst):
        return np.clip((worst - x) / (worst - best) * 100, 0, 100)

    score = (
        .40 * norm_high(df["Download Speed (Mbps)"], 500)
        + .20 * norm_high(df["Upload Speed (Mbps)"], 100)
        + .20 * norm_low(df["Latency (ms)"], 5, 150)
        + .10 * norm_low(df["Jitter (ms)"], 0, 30)
        + .10 * norm_low(df["Ping to Google (ms)"], 5, 150)
    )
    return score.clip(0, 100)


def add_time_features(df):
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out["Hour"] = out["Timestamp"].dt.hour
    out["Date"] = out["Timestamp"].dt.date
    out["Month"] = out["Timestamp"].dt.to_period("M").astype(str)
    return out


def fmt(v, suffix="", digits=1):
    return f"{v:,.{digits}f}{suffix}"


df = load_data()

st.sidebar.markdown("## 📡 5G Intelligence")
st.sidebar.caption("Performance analytics • diagnostics • ML prediction")

if df is None:
    st.warning("No dataset found. Upload a CSV or place it at data/5g_network_data.csv.")
    st.stop()

for issue in validate_dataset(df):
    st.sidebar.warning(issue)

raw = df.copy()
df = preprocess(df)
df = add_time_features(df)
df["Network Quality Score"] = quality_score(df)

# Sidebar controls
st.sidebar.markdown("### Filters")
locations = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()), default=sorted(df["Location"].dropna().unique()))
networks = st.sidebar.multiselect("Network type", sorted(df["Network Type"].dropna().unique()), default=sorted(df["Network Type"].dropna().unique()))
carriers = st.sidebar.multiselect("Carrier", sorted(df["Carrier"].dropna().unique()), default=sorted(df["Carrier"].dropna().unique()))
congestion = st.sidebar.multiselect("Congestion", sorted(df["Network Congestion Level"].dropna().unique()), default=sorted(df["Network Congestion Level"].dropna().unique()))

min_date, max_date = df["Timestamp"].min().date(), df["Timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[
    df["Location"].isin(locations)
    & df["Network Type"].isin(networks)
    & df["Carrier"].isin(carriers)
    & df["Network Congestion Level"].isin(congestion)
].copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[(filtered["Timestamp"].dt.date >= date_range[0]) & (filtered["Timestamp"].dt.date <= date_range[1])]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
if st.sidebar.button("Reset filters", use_container_width=True):
    for key in ["uploaded_csv"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Hero
st.markdown(
    f"""
    <div class="hero">
      <h1>📡 5G Network Intelligence</h1>
      <p>Executive dashboard for network health, carrier performance, congestion diagnostics and ML-powered performance prediction.</p>
      <div style="margin-top:12px"><span class="pill">{len(filtered):,} filtered records</span><span class="pill">{len(df):,} total records</span><span class="pill">{df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload area tucked into sidebar
with st.sidebar.expander("📂 Use another dataset"):
    uploaded = st.file_uploader("CSV file", type=["csv"], key="uploader")
    if uploaded is not None:
        st.session_state["uploaded_csv"] = uploaded
        st.rerun()

# KPIs
avg_dl = filtered["Download Speed (Mbps)"].mean()
avg_ul = filtered["Upload Speed (Mbps)"].mean()
avg_lat = filtered["Latency (ms)"].mean()
avg_signal = filtered["Signal Strength (dBm)"].mean()
avg_jitter = filtered["Jitter (ms)"].mean()
quality = filtered["Network Quality Score"].mean()

k = st.columns(6)
k[0].metric("Records", f"{len(filtered):,}")
k[1].metric("Download", fmt(avg_dl, " Mbps"))
k[2].metric("Upload", fmt(avg_ul, " Mbps"))
k[3].metric("Latency", fmt(avg_lat, " ms"))
k[4].metric("Signal", fmt(avg_signal, " dBm"))
k[5].metric("Quality score", fmt(quality, "/100"))

# Status strip
status = "Excellent" if quality >= 90 else "Very Good" if quality >= 75 else "Good" if quality >= 60 else "Average" if quality >= 40 else "Poor"
st.success(f"Network health: **{status}** — average operational quality score is **{quality:.1f}/100**.")

# Main analytics
st.markdown('<div class="section-title">Performance snapshot</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    trend = filtered.set_index("Timestamp")["Download Speed (Mbps)"].resample("D").mean().reset_index()
    fig = px.line(trend, x="Timestamp", y="Download Speed (Mbps)", title="Daily average download speed", markers=False)
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=50,b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    by_net = filtered.groupby("Network Type", as_index=False).agg(
        Download=("Download Speed (Mbps)", "mean"),
        Latency=("Latency (ms)", "mean"),
        Quality=("Network Quality Score", "mean"),
    )
    fig = px.bar(by_net, x="Network Type", y="Download", title="Average download speed by network type", text_auto=".0f")
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=50,b=10), yaxis_title="Mbps")
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    carrier = filtered.groupby("Carrier", as_index=False).agg(
        Download=("Download Speed (Mbps)", "mean"),
        Latency=("Latency (ms)", "mean"),
        Quality=("Network Quality Score", "mean"),
    ).sort_values("Quality", ascending=False)
    fig = px.bar(carrier, x="Quality", y="Carrier", orientation="h", title="Carrier network quality ranking", text=carrier["Quality"].round(1))
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10), xaxis_title="Quality score / 100")
    st.plotly_chart(fig, use_container_width=True)
with c4:
    loc = filtered.groupby("Location", as_index=False).agg(
        Download=("Download Speed (Mbps)", "mean"),
        Latency=("Latency (ms)", "mean"),
        Signal=("Signal Strength (dBm)", "mean"),
    )
    fig = px.scatter(loc, x="Signal", y="Download", size="Download", color="Latency", hover_name="Location", title="Location: signal strength vs download speed")
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Network diagnostics</div>', unsafe_allow_html=True)
c5, c6, c7 = st.columns(3)
with c5:
    cong = filtered.groupby("Network Congestion Level", as_index=False)[["Download Speed (Mbps)", "Latency (ms)"]].mean()
    fig = px.bar(cong, x="Network Congestion Level", y="Download Speed (Mbps)", title="Congestion vs throughput", text_auto=".0f")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)
with c6:
    fig = px.scatter(filtered.sample(min(5000, len(filtered)), random_state=42), x="Signal Strength (dBm)", y="Download Speed (Mbps)", color="Network Type", opacity=.65, title="Signal strength vs download speed")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)
with c7:
    band = filtered.groupby("Band", as_index=False).agg(Quality=("Network Quality Score", "mean"), Download=("Download Speed (Mbps)", "mean"))
    fig = px.bar(band.sort_values("Quality", ascending=False), x="Band", y="Quality", title="5G band quality", text_auto=".1f")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=50,b=10), yaxis_title="Quality / 100")
    st.plotly_chart(fig, use_container_width=True)

# ML section on home dashboard
st.markdown('<div class="section-title">🤖 ML prediction</div>', unsafe_allow_html=True)
ml_left, ml_right = st.columns([1, 1.7])
with ml_left:
    target = st.selectbox("Prediction target", PREDICTION_TARGETS, index=PREDICTION_TARGETS.index(DEFAULT_TARGET))
    test_size = st.slider("Test size", .1, .4, .2, .05)
    feature_cols = default_feature_columns(filtered, target)
    st.caption(f"Using {len(feature_cols)} predictive inputs while excluding outcome-like fields to reduce leakage.")
    train = st.button("Train ML models", type="primary", use_container_width=True)

if train:
    with st.spinner("Training four regression models…"):
        results = train_and_evaluate(filtered, feature_cols, target, test_size)
    st.session_state["ml_results"] = results
    st.session_state["ml_target"] = target

with ml_right:
    results = st.session_state.get("ml_results")
    if results and st.session_state.get("ml_target") == target:
        best = best_model_name(results)
        rows = []
        for name, r in results.items():
            rows.append({"Model": name, "MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"]})
        metrics = pd.DataFrame(rows).sort_values("R²", ascending=False)
        st.dataframe(metrics.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.3f}"}), use_container_width=True, hide_index=True)
        st.info(f"Best model: **{best}** with R² = **{results[best]['r2']:.3f}**.")
    else:
        st.info("Choose a target and click **Train ML models** to compare models and unlock prediction inputs.")

# Prediction form
results = st.session_state.get("ml_results")
if results and st.session_state.get("ml_target") == target:
    best = best_model_name(results)
    bundle = results[best]
    st.markdown(f"### 🔮 Predict {target}")
    input_values = {}
    cols = st.columns(3)
    for i, col in enumerate(bundle["feature_cols"]):
        s = df[col]
        with cols[i % 3]:
            if col == "Timestamp":
                input_values[col] = pd.Timestamp.now()
            elif pd.api.types.is_numeric_dtype(s):
                lo, hi = float(s.min()), float(s.max())
                default = float(s.median())
                step = max((hi - lo) / 100, 0.01)
                input_values[col] = st.number_input(col, min_value=lo, max_value=hi, value=default, step=step)
            elif pd.api.types.is_bool_dtype(s):
                input_values[col] = st.checkbox(col, value=bool(s.mode().iloc[0]))
            else:
                opts = sorted(s.dropna().astype(str).unique().tolist())
                input_values[col] = st.selectbox(col, opts)

    if st.button("Generate prediction", type="primary"):
        prediction = predict(bundle, input_values)
        st.metric(f"Predicted {target}", f"{prediction:.2f}")

        imp = feature_importance(bundle)
        if not imp.empty:
            fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top model features")
            fig.update_layout(height=430, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

# Footer / data table
with st.expander("🔎 Filtered data preview"):
    st.dataframe(filtered.drop(columns=["Date"], errors="ignore").head(500), use_container_width=True, height=420)

st.caption("5G Network Intelligence • Analytics dashboard with ML-assisted performance prediction")
