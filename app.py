import os

import streamlit as st

from utils.data_utils import dataset_summary, load_csv, preprocess, validate_dataset

st.set_page_config(page_title="5G Network Analytics", page_icon="📡", layout="wide")

st.title("📡 5G Network Performance Analysis, Speed Testing & ML Prediction")
st.caption(
    "Analyze historical 5G network data, run live speed tests, and predict "
    "network performance using machine learning."
)

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload 5G network dataset (CSV)", type=["csv"])

DEFAULT_PATH = os.path.join("data", "5g_network_data.csv")

if uploaded is not None:
    st.session_state["df"] = load_csv(uploaded)
elif "df" not in st.session_state and os.path.exists(DEFAULT_PATH):
    st.session_state["df"] = load_csv(DEFAULT_PATH)

df = st.session_state.get("df")

if df is None:
    st.info(
        "No dataset loaded yet. Upload a CSV from the sidebar, or place a file at "
        f"`{DEFAULT_PATH}`, to unlock the analytics, EDA, correlation, and ML pages.\n\n"
        "The **Live Speed Test** page works independently of the dataset."
    )
else:
    for issue in validate_dataset(df):
        st.warning(issue)

    st.session_state["df_clean"] = preprocess(df)
    summary = dataset_summary(st.session_state["df_clean"])

    st.subheader("Dashboard Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{summary['rows']:,}")
    c2.metric("Columns", summary["columns"])
    c3.metric("Missing Values (raw)", summary["missing_values"])
    c4.metric("Duplicate Rows Removed", summary["duplicate_rows"])

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.write("**Numeric columns**")
    col1.write(summary["numeric_columns"] or "None detected")
    col2.write("**Categorical columns**")
    col2.write(summary["categorical_columns"] or "None detected")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Use the pages in the sidebar to explore performance analytics, run a live "
    "speed test, and train/compare ML models."
)
