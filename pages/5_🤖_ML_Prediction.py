import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_utils import get_dataframe
from utils.ml_utils import (
    DEFAULT_TARGET,
    PREDICTION_TARGETS,
    best_model_name,
    default_feature_columns,
    feature_importance,
    predict,
    train_and_evaluate,
)

st.set_page_config(page_title="ML Performance Prediction", page_icon="🤖", layout="wide")
st.title("🤖 ML Network Performance Prediction")
st.caption("Predict network performance from signal, device, carrier, band, congestion and time features.")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

numeric_targets = [c for c in PREDICTION_TARGETS if c in df.columns]
if not numeric_targets:
    numeric_targets = df.select_dtypes(include="number").columns.tolist()

st.subheader("1. Choose prediction target")
target_col = st.selectbox(
    "What should the ML model predict?",
    numeric_targets,
    index=numeric_targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in numeric_targets else 0,
    help="Download Speed is recommended for this dataset. Dropped Connection is intentionally not used because it behaves almost randomly in this dataset.",
)

all_candidates = [c for c in df.columns if c != target_col]
defaults = [c for c in default_feature_columns(df, target_col) if c in all_candidates]
feature_cols = st.multiselect(
    "Prediction inputs",
    all_candidates,
    default=defaults,
    help="Categorical variables such as carrier, location and band are automatically one-hot encoded. Timestamp is converted to hour/day/month.",
)

c1, c2 = st.columns(2)
c1.metric("Training records", f"{len(df):,}")
c2.metric("Selected inputs", len(feature_cols))

if not feature_cols:
    st.warning("Select at least one prediction input.")
    st.stop()

if st.button("🚀 Train ML Models", type="primary"):
    with st.spinner("Training four regression models and evaluating them on a held-out test set..."):
        results = train_and_evaluate(df, feature_cols, target_col)
    st.session_state["ml_results"] = results
    st.session_state["ml_target"] = target_col
    st.session_state["ml_features"] = feature_cols
    st.success("Training complete.")

if "ml_results" not in st.session_state or st.session_state.get("ml_target") != target_col:
    st.info("Click **Train ML Models** to generate predictions and model metrics.")
    st.stop()

results = st.session_state["ml_results"]
best = best_model_name(results)
r = results[best]

st.subheader("2. Model performance")
m1, m2, m3 = st.columns(3)
m1.metric("Best model", best)
m2.metric("R²", f"{r['r2']:.3f}")
m3.metric("RMSE", f"{r['rmse']:.2f}")

rows = [{"Model": n, "MAE": v["mae"], "RMSE": v["rmse"], "R²": v["r2"]} for n, v in results.items()]
comparison = pd.DataFrame(rows).sort_values("R²", ascending=False)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("3. Most influential features")
imp = feature_importance(r)
if not imp.empty:
    fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title=f"Top features — {best}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Feature importance is available for the tree-based models.")

st.subheader("4. Make a new prediction")
input_values = {}
cols = st.columns(3)
for i, col in enumerate(feature_cols):
    target_widget = cols[i % 3]
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        input_values[col] = target_widget.selectbox(col, [False, True], index=0, key=f"pred_{col}")
    elif pd.api.types.is_numeric_dtype(series):
        lo, hi = float(series.min()), float(series.max())
        default = float(series.median())
        input_values[col] = target_widget.number_input(col, min_value=lo, max_value=hi, value=default, key=f"pred_{col}")
    else:
        options = series.dropna().astype(str).unique().tolist()
        input_values[col] = target_widget.selectbox(col, options, key=f"pred_{col}")

if st.button("🔮 Predict Network Performance", type="primary"):
    prediction = predict(r, input_values)
    st.success(f"Predicted **{target_col}**: **{prediction:,.2f}**")
