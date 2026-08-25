import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="🧠", layout="wide")
st.title("🧠 Model Comparison")

results = st.session_state.get("ml_results")
target = st.session_state.get("ml_target", "target")
if not results:
    st.warning("Train models on the ML Performance Prediction page first.")
    st.stop()

rows = [{"Model": name, "MAE": r["mae"], "MSE": r["mse"], "RMSE": r["rmse"], "R²": r["r2"]} for name, r in results.items()]
comparison = pd.DataFrame(rows).sort_values("R²", ascending=False)
best = comparison.iloc[0]["Model"]

st.caption(f"Target: **{target}**")
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.success(f"Best performing model: **{best}**")

fig = px.bar(comparison, x="Model", y="R²", title=f"R² Score — Predicting {target}", text_auto=".3f")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(comparison, x="Model", y="RMSE", title="RMSE comparison", text_auto=".2f")
st.plotly_chart(fig2, use_container_width=True)
