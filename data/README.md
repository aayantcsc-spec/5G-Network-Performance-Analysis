# Dataset folder

Place your 5G network dataset here as:

```
data/5g_network_data.csv
```

`app.py` auto-loads it from this path on startup (or you can upload a CSV
from the sidebar instead). `data/speed_test_history.csv` is created and
appended to automatically by the Live Speed Test page — no action needed.

No columns are hard-coded: the EDA, Correlation, ML Prediction, and Location
Analysis pages inspect whatever numeric/categorical/location columns are
actually present and build their controls from that.

## Candidate public 5G datasets

Shared for reference while picking a dataset — not yet reviewed or downloaded:

- https://ieee-dataport.org/documents/5g-traffic-datasets
- https://western-oc2-lab.github.io/5G-Core-Networks-Datasets/
- https://github.com/DLTeamTUC/5GDatasets
- https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets
- https://arxiv.org/pdf/2301.09201v1

Before committing to one, check that it has at least a handful of numeric
performance columns (throughput/latency/signal-type metrics) so the EDA,
correlation, and ML pages have something meaningful to work with.
