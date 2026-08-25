import numpy as np

QUALITY_BANDS = [
    (90, "Excellent"),
    (75, "Very Good"),
    (60, "Good"),
    (40, "Average"),
    (0, "Poor"),
]


def _normalize(value, best, worst):
    """Linearly map value to 0-100 where `best` -> 100 and `worst` -> 0.

    `best` may be numerically lower than `worst` for metrics where lower is
    better (ping, jitter, packet loss) -- the interpolation direction just
    flips automatically.
    """
    if best == worst:
        return 100.0
    score = (value - worst) / (best - worst) * 100
    return float(np.clip(score, 0, 100))


def compute_quality_score(download_mbps, upload_mbps, ping_ms, jitter_ms, packet_loss_pct):
    """Weighted network quality score (0-100).

    Weights: 40% download, 20% upload, 20% ping, 10% jitter, 10% packet loss.
    Reference ranges (500 Mbps / 100 Mbps / 150 ms / 30 ms / 10%) are
    illustrative caps for a typical 5G connection, not a telecom standard --
    tune them here if your project needs different bounds.
    """
    download_score = _normalize(download_mbps, best=500, worst=0)
    upload_score = _normalize(upload_mbps, best=100, worst=0)
    ping_score = _normalize(ping_ms, best=5, worst=150)
    jitter_score = _normalize(jitter_ms, best=0, worst=30)
    packet_loss_score = _normalize(packet_loss_pct, best=0, worst=10)

    return (
        0.4 * download_score
        + 0.2 * upload_score
        + 0.2 * ping_score
        + 0.1 * jitter_score
        + 0.1 * packet_loss_score
    )


def classify_quality(score: float) -> str:
    for threshold, label in QUALITY_BANDS:
        if score >= threshold:
            return label
    return "Poor"
