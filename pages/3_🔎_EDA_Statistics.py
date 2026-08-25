import plotly.express as px
import streamlit as st

from utils.data_utils import get_dataframe

st.set_page_config(page_title="EDA & Statistics", page_icon="🔎", layout="wide")
st.title("🔎 Exploratory Data Analysis")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

st.subheader("Descriptive Statistics")
st.dataframe(df.describe().T, use_container_width=True)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    col = st.selectbox("Select a column to inspect", numeric_cols)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(px.box(df, y=col, title=f"Outlier check: {col}"), use_container_width=True)

st.subheader("Missing Values")
missing = df.isna().sum()
missing = missing[missing > 0]
if missing.empty:
    st.success("No missing values detected.")
else:
    st.dataframe(missing.rename("missing_count"), use_container_width=True)

st.subheader("Duplicate Rows")
st.write(f"{df.duplicated().sum()} duplicate rows found.")
