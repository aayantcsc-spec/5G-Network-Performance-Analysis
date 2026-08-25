import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_utils import dataset_summary, load_csv, preprocess, validate_dataset

st.set_page_config(page_title="5G Network Data Analyzer", page_icon="📡", layout="wide")

st.title("📡 5G network data analyzer")
st.caption("Upload a CSV file and it's analyzed automatically — no columns are hard-coded.")

DEFAULT_PATH = ROOT / "data" / "5g_network_data.csv"

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    df = load_csv(uploaded)
elif DEFAULT_PATH.exists():
    df = load_csv(DEFAULT_PATH)
    st.caption(f"No file uploaded — showing the sample dataset (`{DEFAULT_PATH.name}`).")
else:
    df = None

if df is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

for issue in validate_dataset(df):
    st.warning(issue)

df = preprocess(df)
st.session_state["df"] = df
st.session_state["df_clean"] = df

summary = dataset_summary(df)

st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['rows']:,}")
c2.metric("Columns", summary["columns"])
c3.metric("Missing values", f"{summary['missing_values']:,}")
c4.metric("Duplicate rows", f"{summary['duplicate_rows']:,}")

st.subheader("Preview")
st.dataframe(df.head(50), width="stretch")

st.subheader("Summary statistics")
st.dataframe(df.describe(include="all").transpose(), width="stretch")

numeric_cols = summary["numeric_columns"]
if numeric_cols:
    st.subheader("Chart a column")
    column = st.selectbox("Numeric column", numeric_cols)
    st.line_chart(df[column])

st.download_button(
    "Download this data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="analyzed_data.csv",
    mime="text/csv",
)
