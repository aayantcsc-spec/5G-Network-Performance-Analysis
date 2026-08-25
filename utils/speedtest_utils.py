import os
import platform
import re
import statistics
import subprocess
from datetime import datetime

import pandas as pd

HISTORY_PATH = os.path.join("data", "speed_test_history.csv")
PING_HOST = "8.8.8.8"
PING_COUNT = 8

HISTORY_COLUMNS = [
    "timestamp", "download_mbps", "upload_mbps", "ping_ms",
    "jitter_ms", "packet_loss_pct", "quality_score", "quality",
]


def _ping_stats(host: str = PING_HOST, count: int = PING_COUNT) -> dict:
    """Ping via the OS `ping` command and parse latency/jitter/packet loss.

    Uses the system command instead of a raw-socket library so it runs
    without admin/root privileges on Windows, macOS, or Linux.
    """
    is_windows = platform.system().lower() == "windows"
    cmd = ["ping", "-n" if is_windows else "-c", str(count), host]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=count * 3 + 10)
        output = result.stdout
    except Exception as exc:
        raise RuntimeError(f"Ping measurement failed: {exc}") from exc

    times = [float(m) for m in re.findall(r"time[=<]\s*([\d.]+)\s*ms", output, re.IGNORECASE)]

    loss_match = re.search(r"([\d.]+)\s*%\s*(?:packet )?loss", output, re.IGNORECASE)
    if loss_match:
        loss_pct = float(loss_match.group(1))
    else:
        loss_pct = max(0.0, (count - len(times)) / count * 100)

    if not times:
        raise RuntimeError("No successful ping replies received; check network connectivity.")

    return {
        "ping_ms": round(statistics.mean(times), 2),
        "jitter_ms": round(statistics.pstdev(times), 2) if len(times) > 1 else 0.0,
        "packet_loss_pct": round(loss_pct, 2),
    }


def run_speed_test() -> dict:
    """Run a real internet speed test (download/upload) plus a ping sweep
    for latency, jitter, and packet loss."""
    import speedtest

    client = speedtest.Speedtest(secure=True)
    client.get_best_server()
    download_bps = client.download()
    upload_bps = client.upload()

    ping_stats = _ping_stats()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "download_mbps": round(download_bps / 1_000_000, 2),
        "upload_mbps": round(upload_bps / 1_000_000, 2),
        **ping_stats,
    }


def save_result(result: dict, quality_label: str, quality_score: float) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    row = {**result, "quality_score": round(quality_score, 1), "quality": quality_label}
    write_header = not os.path.exists(HISTORY_PATH)
    pd.DataFrame([row])[HISTORY_COLUMNS].to_csv(
        HISTORY_PATH, mode="a", header=write_header, index=False
    )


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(HISTORY_PATH)
