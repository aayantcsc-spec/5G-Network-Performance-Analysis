import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Actual vs Predicted", page_icon="📈", layout="wide")
st.title("📈 Actual vs Predicted")

results = st.session_state.get("ml_results")
target = st.session_state.get("ml_target", "target")
if not results:
    st.warning("Train models on the ML Performance Prediction page first.")
    st.stop()

model_name = st.selectbox("Select model", list(results.keys()))
r = results[model_name]

comp_df = pd.DataFrame({"Actual": r["y_test"], "Predicted": r["y_pred"]})
fig = px.scatter(comp_df, x="Actual", y="Predicted", trendline="ols", title=f"{model_name}: Actual vs Predicted {target}")
min_v, max_v = comp_df.min().min(), comp_df.max().max()
fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="gray"))
st.plotly_chart(fig, use_container_width=True)

comp_df["Residual"] = comp_df["Actual"] - comp_df["Predicted"]
st.subheader("Residuals")
st.plotly_chart(px.histogram(comp_df, x="Residual", title="Residual distribution"), use_container_width=True)
