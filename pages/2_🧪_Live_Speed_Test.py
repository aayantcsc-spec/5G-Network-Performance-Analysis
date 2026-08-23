import plotly.express as px
import streamlit as st

from utils.quality_utils import classify_quality, compute_quality_score
from utils.speedtest_utils import load_history, run_speed_test, save_result

st.set_page_config(page_title="Live Speed Test", page_icon="🧪", layout="wide")
st.title("🧪 Live Speed Test")
st.caption("Measures your current internet connection's download, upload, ping, jitter, and packet loss.")

if st.button("▶ Start Test", type="primary"):
    with st.spinner("Running speed test... this can take up to a minute."):
        try:
            result = run_speed_test()
        except Exception as exc:
            st.error(f"Speed test failed: {exc}")
            result = None

    if result:
        score = compute_quality_score(
            result["download_mbps"], result["upload_mbps"],
            result["ping_ms"], result["jitter_ms"], result["packet_loss_pct"],
        )
        quality = classify_quality(score)
        save_result(result, quality, score)
        st.session_state["last_speed_test"] = {**result, "quality_score": round(score, 1), "quality": quality}

if "last_speed_test" in st.session_state:
    r = st.session_state["last_speed_test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Download", f"{r['download_mbps']} Mbps")
    c2.metric("Upload", f"{r['upload_mbps']} Mbps")
    c3.metric("Ping", f"{r['ping_ms']} ms")
    c4.metric("Packet Loss", f"{r['packet_loss_pct']}%")
    st.metric("Network Quality", f"{r['quality']} — {r['quality_score']}%")
else:
    st.info("Click **Start Test** to measure your current network performance.")

st.markdown("---")
st.subheader("Speed Test History")
history = load_history()
if history.empty:
    st.write("No speed tests recorded yet.")
else:
    st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True)
    fig = px.line(history, x="timestamp", y=["download_mbps", "upload_mbps"], title="Speed trend over time")
    st.plotly_chart(fig, use_container_width=True)
