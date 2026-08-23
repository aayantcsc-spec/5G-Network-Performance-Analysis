# Product Requirements Document (PRD)

## 1. Product Title

**5G Network Performance Analysis, Speed Testing & Machine Learning-Based Performance Prediction System**

## 2. Product Overview

The proposed system is a web-based Data Science application that analyzes 5G network performance data, provides an interactive dashboard, performs internet speed tests, evaluates network quality, and uses Machine Learning to predict network performance.

The system will help users understand important network parameters such as **download speed, upload speed, latency, jitter, packet loss, signal strength, bandwidth, and network utilization**.

Historical 5G network data will be analyzed using Exploratory Data Analysis (EDA), statistical techniques, visualization, and Machine Learning models. A Streamlit-based dashboard will provide an easy-to-use interface for viewing results and generating predictions.

---

## 3. Problem Statement

5G networks generate large amounts of performance data. Understanding this data manually is difficult because network performance depends on several factors, including:

* Signal strength
* Network traffic
* Number of connected users
* Bandwidth
* Latency
* Packet loss
* Download/upload speed
* Network location

There is a need for a system that can **analyze network performance, conduct speed tests, visualize network conditions, and predict future/expected performance using Machine Learning**.

---

## 4. Product Objectives

### Primary objectives

1. Analyze historical 5G network performance data.
2. Perform comprehensive EDA.
3. Provide an interactive network analytics dashboard.
4. Perform real-time internet speed testing.
5. Calculate network quality based on multiple parameters.
6. Train Machine Learning models for performance prediction.
7. Compare different ML algorithms.
8. Display actual vs predicted performance.
9. Maintain speed-test history.
10. Provide actionable insights about network performance.

---

## 5. Target Users

### 5.1 Students / Researchers

Can use the system for:

* Data analysis
* Academic research
* Network-performance studies
* ML experimentation

### 5.2 Network Analysts

Can analyze:

* Network quality
* Performance trends
* Latency
* Throughput
* Packet loss

### 5.3 General Users

Can use the Speed Test module to understand their current internet performance.

### 5.4 Network Administrators

Can use analytics and prediction results to identify potential performance problems.

---

## 6. Product Scope

### In Scope

* Dataset upload
* Data preprocessing
* EDA
* Statistical analysis
* Interactive charts
* Network performance dashboard
* Speed testing
* Network quality scoring
* ML model training
* Model comparison
* Performance prediction
* Actual vs predicted visualization
* Speed-test history
* Location-based analysis where location information exists

### Out of Scope

* Controlling physical 5G network infrastructure
* Changing network configuration
* Mobile tower management
* Telecom billing
* SIM management
* Direct network optimization at telecom-provider level

---

## 7. Major Features

### 7.1 Dashboard Overview

The dashboard should display:

* Total records
* Average download speed
* Average upload speed
* Average latency
* Average signal strength
* Average packet loss
* Overall network quality

---

## 8. Live Speed Test

The system provides a **Start Speed Test** button that measures:

* Download speed
* Upload speed
* Ping
* Latency
* Jitter
* Packet loss

The system records the test result with a timestamp.

> The exact measurements available depend on the speed-testing technology/API used.

---

## 9. Network Quality Score

The system generates a network-quality score from relevant metrics:

```text
90–100 → Excellent
75–89  → Very Good
60–74  → Good
40–59  → Average
0–39   → Poor
```

The scoring formula is documented and configurable rather than treated as a telecom-standard score (see `utils/quality_utils.py`).

---

## 10. EDA Module

### Univariate Analysis

Mean, median, mode, standard deviation, min/max, quartiles, distribution, outliers.

### Bivariate Analysis

Examples: signal strength vs download speed, number of users vs latency, packet loss vs throughput, bandwidth vs throughput.

### Multivariate Analysis

Correlation matrix, feature relationships, network-performance patterns.

---

## 11. Visualization Module

Bar charts, line charts, scatter plots, histograms, box plots, correlation heatmaps, KPI cards, interactive Plotly charts.

---

## 12. Machine Learning Module

### Candidate models

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression

The final model is selected based on evaluation metrics: MAE, MSE, RMSE, R² Score.

---

## 13. ML Prediction

The user enters network parameters (determined from the actual dataset columns) and the system generates a predicted value plus network quality classification.

---

## 14. Model Comparison

The dashboard displays a metrics table across all trained models and highlights the best-performing one.

---

## 15. Speed Test History

The system maintains previous speed tests and a speed trend chart showing whether network performance is improving or declining.

---

## 16. Location Analysis

If the dataset contains location information, the system displays performance by location, and a map when latitude/longitude data is available.

---

## 17. Functional Requirements

| ID    | Requirement                                               |
| ----- | --------------------------------------------------------- |
| FR-01 | System shall load 5G network data.                        |
| FR-02 | System shall validate uploaded data.                      |
| FR-03 | System shall preprocess the dataset.                      |
| FR-04 | System shall calculate descriptive statistics.            |
| FR-05 | System shall display interactive visualizations.          |
| FR-06 | System shall perform speed tests.                         |
| FR-07 | System shall calculate network-quality scores.            |
| FR-08 | System shall train ML models.                             |
| FR-09 | System shall compare ML models.                           |
| FR-10 | System shall predict network performance.                 |
| FR-11 | System shall display actual vs predicted results.         |
| FR-12 | System shall maintain speed-test history.                 |
| FR-13 | System shall provide location analysis where data exists. |

---

## 18. Non-Functional Requirements

**Performance** — dashboard loads efficiently; data processing optimized for the project dataset; charts respond quickly to filters.

**Usability** — simple navigation, clear graphs, readable KPI cards, beginner-friendly interface.

**Reliability** — handles invalid datasets gracefully, displays meaningful error messages, prevents prediction when required input is missing.

**Security** — validates uploaded files, restricts accepted file types, avoids exposing sensitive information.

**Maintainability** — modular Python code, separate preprocessing/ML/dashboard components, documented functions.

---

## 19. Technology Stack

| Layer             | Technology                  |
| ----------------- | --------------------------- |
| Programming       | Python                      |
| Dashboard         | Streamlit                   |
| Data Processing   | Pandas, NumPy               |
| Visualization     | Plotly, Matplotlib, Seaborn |
| ML                | Scikit-learn                |
| Model Storage     | Joblib/Pickle               |
| Dataset           | CSV                         |
| Optional Database | MySQL/SQLite                |
| Development       | VS Code / Jupyter Notebook  |
| Version Control   | Git/GitHub                  |

---

## 20. High-Level Architecture

```text
                  ┌────────────────────┐
                  │   5G Dataset       │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │ Data Preprocessing │
                  └─────────┬──────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       ┌─────────────┐             ┌─────────────┐
       │     EDA     │             │ Visualization│
       └──────┬──────┘             └──────┬──────┘
              │                           │
              └─────────────┬─────────────┘
                            ↓
                   ┌────────────────┐
                   │ Machine        │
                   │ Learning       │
                   └───────┬────────┘
                           ↓
                   ┌────────────────┐
                   │ Prediction     │
                   └───────┬────────┘
                           ↓
                  ┌──────────────────┐
                  │ Streamlit        │
                  │ Dashboard        │
                  └──────────────────┘

        ┌─────────────────────────────┐
        │       Live Speed Test       │
        └──────────────┬──────────────┘
                       ↓
              ┌────────────────┐
              │ Quality Score  │
              └────────────────┘
```

---

## Future Enhancements

* Android/mobile application
* Real-time network coverage map
* GPS-based performance mapping
* Poor-network alerts
* Cloud deployment
* Real-time telecom data integration
* Time-series forecasting
* Deep Learning models
* Multi-user accounts
* Automatic PDF performance reports

### Final project positioning

> A Data Science and Machine Learning-based 5G Network Analytics platform that combines historical network-data analysis, interactive visualization, real-time speed testing, network-quality assessment, and predictive performance modeling in a unified Streamlit dashboard.
