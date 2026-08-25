import plotly.express as px
import streamlit as st

from utils.data_utils import get_dataframe

st.set_page_config(page_title="Correlation Analysis", page_icon="🔥", layout="wide")
st.title("🔥 Correlation Analysis")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

numeric_df = df.select_dtypes(include="number")
if numeric_df.shape[1] < 2:
    st.info("Need at least two numeric columns for correlation analysis.")
    st.stop()

corr = numeric_df.corr()
fig = px.imshow(
    corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, title="Correlation Heatmap",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Scatter Explorer")
c1, c2 = st.columns(2)
x_col = c1.selectbox("X axis", numeric_df.columns, index=0)
y_col = c2.selectbox("Y axis", numeric_df.columns, index=min(1, len(numeric_df.columns) - 1))
st.plotly_chart(
    px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}"),
    use_container_width=True,
)
