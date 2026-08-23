import streamlit as st

from utils.data_utils import get_dataframe
from utils.ml_utils import best_model_name, predict, train_and_evaluate

st.set_page_config(page_title="ML Performance Prediction", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Performance Prediction")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) < 2:
    st.info("Need at least two numeric columns (one target, one feature) to train a model.")
    st.stop()

st.subheader("1. Choose target and features")
target_col = st.selectbox("Target column to predict", numeric_cols)
feature_options = [c for c in numeric_cols if c != target_col]
feature_cols = st.multiselect("Feature columns", feature_options, default=feature_options)

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

if st.button("Train Models", type="primary"):
    with st.spinner("Training Linear Regression, Decision Tree, Random Forest, and Gradient Boosting..."):
        results = train_and_evaluate(df, feature_cols, target_col)
    st.session_state["ml_results"] = results
    st.session_state["ml_target"] = target_col
    st.session_state["ml_features"] = feature_cols
    st.success("Training complete. See Model Comparison and Actual vs Predicted pages for details.")

if "ml_results" in st.session_state and st.session_state.get("ml_target") == target_col:
    results = st.session_state["ml_results"]
    best = best_model_name(results)
    st.subheader("2. Predict")
    st.caption(f"Using best model: **{best}** (R² = {results[best]['r2']:.3f})")

    input_values = {}
    cols = st.columns(min(3, len(feature_cols)))
    for i, col in enumerate(feature_cols):
        col_min, col_max = float(df[col].min()), float(df[col].max())
        default = float(df[col].median())
        input_values[col] = cols[i % len(cols)].number_input(
            col, value=default, min_value=col_min, max_value=col_max
        )

    if st.button("Predict"):
        prediction = predict(results[best], input_values)
        st.metric(f"Predicted {target_col}", f"{prediction:.2f}")
