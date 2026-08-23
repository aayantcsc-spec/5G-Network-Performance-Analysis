import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="🧠", layout="wide")
st.title("🧠 Model Comparison")

results = st.session_state.get("ml_results")
if not results:
    st.warning("Train models on the ML Performance Prediction page first.")
    st.stop()

rows = [
    {"Model": name, "MAE": r["mae"], "MSE": r["mse"], "RMSE": r["rmse"], "R2": r["r2"]}
    for name, r in results.items()
]
comparison = pd.DataFrame(rows).set_index("Model")
best = comparison["R2"].idxmax()

st.subheader("Evaluation Metrics")
st.dataframe(
    comparison.style
    .highlight_max(subset=["R2"], color="lightgreen")
    .highlight_min(subset=["MAE", "MSE", "RMSE"], color="lightgreen"),
    use_container_width=True,
)
st.success(f"Best performing model: **{best}** (highest R²)")

fig = px.bar(comparison.reset_index(), x="Model", y="R2", title="R² Score by Model", color="Model")
st.plotly_chart(fig, use_container_width=True)
