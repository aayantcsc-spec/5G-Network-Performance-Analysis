import plotly.express as px
import streamlit as st

from utils.data_utils import get_dataframe

st.set_page_config(page_title="Network Performance", page_icon="📊", layout="wide")
st.title("📊 Network Performance")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

if not numeric_cols:
    st.info("No numeric columns found in the dataset.")
    st.stop()

metric = st.selectbox("Select a performance metric", numeric_cols)

filter_col = None
if categorical_cols:
    choice = st.selectbox("Group / filter by (optional)", ["None"] + categorical_cols)
    filter_col = None if choice == "None" else choice

col1, col2 = st.columns(2)
with col1:
    fig = px.line(df.reset_index(), x="index", y=metric, color=filter_col, title=f"{metric} over records")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.histogram(df, x=metric, color=filter_col, title=f"{metric} distribution")
    st.plotly_chart(fig2, use_container_width=True)

if filter_col:
    st.subheader(f"Average metrics by {filter_col}")
    st.dataframe(df.groupby(filter_col)[numeric_cols].mean().round(2), use_container_width=True)
