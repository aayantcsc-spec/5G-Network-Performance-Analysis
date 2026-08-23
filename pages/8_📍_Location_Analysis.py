import plotly.express as px
import streamlit as st

from utils.data_utils import get_dataframe

st.set_page_config(page_title="Location Analysis", page_icon="📍", layout="wide")
st.title("📍 Network / Location Analysis")

df = get_dataframe()
if df is None:
    st.warning("Load a dataset on the Home page first.")
    st.stop()

location_candidates = [
    c for c in df.columns
    if any(k in c.lower() for k in ["location", "city", "region", "cell", "site", "area"])
]
lat_candidates = [c for c in df.columns if "lat" in c.lower()]
lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not location_candidates and not (lat_candidates and lon_candidates):
    st.info(
        "No location-related columns (e.g. `location`, `city`, `latitude`/`longitude`) "
        "were detected in this dataset."
    )
    st.stop()

if location_candidates:
    loc_col = st.selectbox("Location column", location_candidates)
    metric_cols = st.multiselect("Metrics to compare", numeric_cols, default=numeric_cols[:3])
    if metric_cols:
        summary = df.groupby(loc_col)[metric_cols].mean().round(2)
        st.subheader(f"Average metrics by {loc_col}")
        st.dataframe(summary, use_container_width=True)
        st.plotly_chart(
            px.bar(summary.reset_index(), x=loc_col, y=metric_cols, barmode="group", title="Performance by location"),
            use_container_width=True,
        )

if lat_candidates and lon_candidates:
    st.subheader("Map view")
    lat_col, lon_col = lat_candidates[0], lon_candidates[0]
    choice = st.selectbox("Color points by", ["None"] + numeric_cols)
    color_col = None if choice == "None" else choice
    st.plotly_chart(
        px.scatter_mapbox(
            df, lat=lat_col, lon=lon_col, color=color_col,
            zoom=3, height=500, mapbox_style="open-street-map",
        ),
        use_container_width=True,
    )
