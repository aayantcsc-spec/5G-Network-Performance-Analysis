# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose

This SRS defines the functional and non-functional requirements of the **5G Network Performance Analysis, Speed Testing & Machine Learning-Based Performance Prediction System**.

The document serves as a technical specification for development, testing, and evaluation of the system.

---

## 2. System Description

The system is a Python-based web application that combines:

**Data Science + Machine Learning + Network Speed Testing + Interactive Visualization.**

The system analyzes historical 5G network data and provides performance insights through a Streamlit dashboard.

---

## 3. System Users

### User

Can: view dashboard, run speed test, view results, analyze network statistics, generate predictions.

### Analyst/Administrator

Can additionally: upload datasets, train models, evaluate models, analyze model performance.

---

## 4. System Modules

```text
Module 1  → Dataset Management
Module 2  → Data Preprocessing
Module 3  → EDA
Module 4  → Visualization
Module 5  → Speed Test
Module 6  → Network Quality
Module 7  → Machine Learning
Module 8  → Prediction
Module 9  → Model Evaluation
Module 10 → History
Module 11 → Location Analysis
```

---

## 5. Detailed Functional Requirements

### FR-01 Dataset Management

The system shall accept CSV datasets, validate column structure, display number of rows/columns, display data types, identify missing values, and detect duplicate records.

### FR-02 Data Preprocessing

The system shall handle missing values, remove duplicates, handle outliers where appropriate, encode categorical variables, scale numerical features when required, and prepare data for ML.

### FR-03 EDA

The system shall calculate mean, median, mode, range, standard deviation, quartiles, correlation, and display suitable charts.

### FR-04 Speed Test

The system provides Start Test → Measure → Process → Display, displaying available measurements: download speed, upload speed, ping, latency, jitter, packet loss.

### FR-05 Network Quality

The system calculates a network quality score using defined project criteria and classifies the result into Excellent / Very Good / Good / Average / Poor.

### FR-06 Machine Learning

The system shall split data into training and testing sets, train selected ML models, generate predictions, calculate evaluation metrics, compare models, and select the best model.

### FR-07 Prediction

The system accepts valid network parameters and generates a predicted performance value. Input fields are determined from the actual dataset.

### FR-08 Model Evaluation

The system calculates MAE, MSE, RMSE, R² and presents results in a comparison table and visualization.

### FR-09 Historical Analysis

The system stores/displays speed-test results including date, time, download speed, upload speed, latency, jitter, packet loss, and quality score.

### FR-10 Dashboard

The dashboard contains: Home, Performance Analysis, Speed Test, EDA, Correlation, ML Prediction, Model Comparison, Actual vs Predicted, History, Location Analysis.

---

## 6. Use Cases

### Use Case 1 — Run Speed Test

**Actor:** User

```text
User opens dashboard → Selects Speed Test → Clicks Start Test
→ System performs test → System calculates metrics → System displays results
→ System calculates quality → Result is stored/displayed in history
```

### Use Case 2 — Predict Performance

**Actor:** User/Analyst

```text
Enter network parameters → Validate input → Preprocess input
→ Load trained model → Generate prediction → Display predicted performance
```

### Use Case 3 — Analyze Dataset

```text
Upload dataset → Validate dataset → Preprocess → EDA
→ Generate charts → Display insights
```

---

## 7. Data Requirements

The exact dataset schema is finalized after inspecting the actual `5g_network_data.csv`.

Potential fields include: Timestamp, Location, Signal Strength, Bandwidth, Latency, Jitter, Packet Loss, Download Speed, Upload Speed, Connected Users, Network Utilization, Throughput.

**Important:** These columns are not hard-coded into the implementation. The app maps to whatever columns actually exist in the supplied dataset (see `utils/data_utils.py` and the dynamic column pickers on the EDA/ML pages).

---

## 8. ML Pipeline

```text
Raw Dataset → Data Cleaning → Feature Selection → Feature Engineering
→ Train/Test Split → Scaling → Model Training → Model Evaluation
→ Best Model → Save Model → Streamlit Prediction
```

---

## 9. Expected Output

* **Dashboard** — overall network performance
* **Speed Test** — current internet performance
* **EDA** — understanding of historical 5G data
* **ML** — predicted network performance
* **Model Comparison** — best-performing algorithm
* **History** — previous speed-test results
* **Location Analysis** — performance differences between locations where location data exists

---

## 10. Acceptance Criteria

* [ ] Dataset can be loaded successfully.
* [ ] Missing/invalid data is handled.
* [ ] EDA is available.
* [ ] Interactive charts work.
* [ ] Speed-test module works with the selected implementation.
* [ ] Network-quality score is generated.
* [ ] At least two ML algorithms are trained.
* [ ] Models are evaluated using appropriate metrics.
* [ ] Best model can generate predictions.
* [ ] Actual vs predicted visualization is available.
* [ ] Dashboard integrates all major modules.
* [ ] Application handles invalid inputs without crashing.
