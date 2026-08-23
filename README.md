# 5G Network Performance Analysis, Speed Testing & ML Prediction

Streamlit dashboard combining historical 5G network data analysis, a live
internet speed test, and machine-learning performance prediction.

See [docs/PRD.md](docs/PRD.md) and [docs/SRS.md](docs/SRS.md) for the full
product/requirements spec.

## Setup

```bash
pip install -r requirements.txt
```

Place your dataset at `data/5g_network_data.csv` (see [data/README.md](data/README.md)),
or upload a CSV from the app's sidebar at runtime.

## Run

```bash
streamlit run app.py
```

## Structure

```
app.py                          Home page: dataset upload, overview KPIs
pages/
  1_Network_Performance.py      Performance charts & filters
  2_Live_Speed_Test.py          Real download/upload/ping/jitter/loss test + history
  3_EDA_Statistics.py           Descriptive stats, distributions, outliers
  4_Correlation_Analysis.py     Correlation heatmap & scatter explorer
  5_ML_Prediction.py            Train models, predict from user input
  6_Model_Comparison.py         MAE/MSE/RMSE/R2 comparison table & chart
  7_Actual_vs_Predicted.py      Scatter + residuals for a trained model
  8_Location_Analysis.py        Location/lat-lon breakdown (if present)
utils/
  data_utils.py                 Load, validate, preprocess
  quality_utils.py              Network quality scoring/classification
  speedtest_utils.py            Real speed test + ping-based jitter/loss + history CSV
  ml_utils.py                   Train/evaluate/predict across 4 regressors
data/                           Dataset + speed_test_history.csv (generated)
docs/                           PRD.md, SRS.md
```

## Notes

- The ML/EDA/Correlation/Location pages are fully dynamic — they read
  whatever numeric/categorical/location columns exist in the uploaded
  dataset rather than assuming fixed column names.
- The speed test uses `speedtest-cli` for download/upload and the OS `ping`
  command for latency/jitter/packet loss (no admin privileges required).
- Network quality scoring weights and reference ranges are documented in
  `utils/quality_utils.py` — adjust them if your project needs different
  thresholds.
