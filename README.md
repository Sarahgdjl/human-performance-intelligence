# 🧠 Human Performance Intelligence System
> *Decoding the psychology of goal achievement through behavioral data science*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)](https://shap.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-red)](https://facebook.github.io/prophet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

This project is a **4-module end-to-end data science platform** that analyzes habit tracking
and productivity sensor data to answer one central question:

> **What separates people who consistently achieve their goals from those who don't —
> and can we predict burnout before it happens?**

Using two behavioral datasets (50,000 habit logs across 3,000 users + 60,000 daily sensor
readings across 500 students), the system delivers four interconnected analytical modules:
survival curves that reveal when goals die, explainable ML that decodes habit DNA, a
burnout risk index that scores weekly risk 0–100, and a time series forecast of peak
productivity windows.

---

## 🗂 Project Structure

```
human-performance-intelligence/
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb        ← Data cleaning, feature engineering, merging
│   ├── 02_survival_analysis.ipynb    ← Kaplan-Meier + Cox PH model
│   ├── 03_habit_dna.ipynb            ← XGBoost + SHAP explainability
│   ├── 04_burnout_scoring.ipynb      ← Composite BRI + LightGBM prediction
│   └── 05_productivity_forecast.ipynb← Prophet time series forecasting
│
├── dashboard/
│   └── human_performance.pbix        ← 4-page Power BI executive dashboard
│
├── report/
│   └── executive_summary.pdf         ← Non-technical stakeholder summary
│
├── images/                           ← Key visualizations exported as PNG
│
├── model_card.md                     ← Model documentation & limitations
├── requirements.txt                  ← Python dependencies
└── README.md                         ← You are here
```

---

## 📊 Modules at a Glance

| # | Module | Core Technique | Business Question |
|---|--------|---------------|-------------------|
| **01** | Data Pipeline | Pandas · SQL schema | How do we unify messy behavioral data? |
| **02** | Habit Survival | Kaplan-Meier · Cox PH | *When* do people quit their goals? |
| **03** | Habit DNA | XGBoost · SHAP | *What* predicts goal completion? |
| **04** | Burnout Scoring | LightGBM · Composite Index | *Who* is at risk of burning out next week? |
| **05** | Productivity Forecast | Prophet · Time Series | *When* are peak performance windows? |

---

## 🔑 Key Results

### Module 2 — Habit Survival Analysis
- **82.3%** of habit streaks survive past day 1
- **65.7%** survive past day 3
- **56.0%** survive past day 7
- Users **with accountability partners** have measurably higher 7-day survival
- Cox PH model identifies `user_level`, `difficulty`, and `day_of_week` as top quit-risk factors
- Survival curves differ **significantly across habit categories** (log-rank p < 0.05)

### Module 3 — Habit DNA (XGBoost + SHAP)
- XGBoost achieves **AUC = 0.99** on habit completion prediction
- **SHAP beeswarm** reveals `xp_efficiency` and `user_avg_completion` as dominant drivers
- **4 behavioral archetypes** identified by clustering on SHAP value profiles
- Local waterfall plots explain individual predictions — high confidence, low confidence, borderline

### Module 4 — Burnout Risk Scoring
- **Burnout Risk Index (BRI):** composite 0–100 score built from 8 behavioral signals
- BRI during **deadline weeks** is on average **+8–12 points higher** than normal weeks
- LightGBM **predicts next-week high-burnout** with AUC > 0.80
- Temporal train/test split (weeks 1–14 train, 15–18 test) prevents future data leakage

### Module 5 — Productivity Forecasting
- Prophet decomposes productivity into **trend + weekly + daily seasonality**
- Forecast horizon: **7 days** with confidence intervals
- Identifies individual **peak performance windows** per student

---

## 🛠 Technical Stack

| Category | Tools |
|---|---|
| **Data manipulation** | Pandas, NumPy |
| **Machine learning** | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (TreeExplainer) |
| **Survival analysis** | Lifelines (KaplanMeierFitter, CoxPHFitter) |
| **Time series** | Prophet |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Power BI (DAX measures) |
| **Notebook format** | Jupyter Notebooks |

---

## 💡 Notable Technical Decisions

### Why Survival Analysis for Habit Streaks?
Most analysts would compute "average streak length." That's wrong — it ignores **censored
data** (streaks still ongoing when data was collected). Kaplan-Meier handles censoring
correctly. This technique is borrowed from clinical trial research and applying it to
behavioral data is a genuine cross-domain insight.

### Why a Temporal Train/Test Split?
For the burnout prediction model, I used weeks 1–14 for training and weeks 15–18 for
testing — not a random split. Random splitting on time series data leaks future
information into training, producing falsely optimistic metrics. The temporal split
reflects real deployment conditions.

### Data Leakage Caught During EDA
The `Complain` column showed near-perfect correlation (r ≈ 0.99) with the churn target.
This would have made the model "cheat" in training and fail in production. It was
identified during the audit phase and dropped before any modeling.

### SHAP on Behavioral Segments
Beyond global feature importance, I clustered users by their **SHAP value profiles** —
not their raw features. This reveals behavioral archetypes based on *how the model
reasons about them*, which is a more nuanced segmentation than standard K-Means on raw data.

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/human-performance-intelligence.git
cd human-performance-intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run notebooks in order**
```bash
jupyter notebook
```
Open and run notebooks `01` through `05` sequentially.
Each notebook saves its outputs to `data/processed/` for the next to consume.

> **Note:** The datasets in `data/raw/` are synthetically generated to mirror
> real-world behavioral data with realistic noise, missing values, and outliers.
> See `notebooks/01_data_pipeline.ipynb` for the generation logic and cleaning pipeline.

---

## 📁 Data

| Dataset | Rows | Description |
|---------|------|-------------|
| `habitica_raw.csv` | 51,500 | Habit logs: streaks, completions, XP, categories |
| `studentlife_raw.csv` | 61,500 | Daily sensor readings: sleep, stress, focus, mood |

Both datasets are **synthetically generated** with realistic noise:
- Mixed date formats, category typos, impossible values (age, sleep hours)
- ~3–8% missing values per column
- ~3% duplicate rows (simulating sync errors)
- Stress scale inconsistency (1–10 vs 1–100)

All issues are documented and resolved in `01_data_pipeline.ipynb`.

---

## 📈 Dashboard

The Power BI dashboard (`dashboard/human_performance.pbix`) contains 4 pages:

| Page | Content |
|------|---------|
| **Overview** | BRI distribution, risk tier KPIs, semester trajectory |
| **Survival Analysis** | KM curves by category, difficulty, accountability |
| **Habit DNA** | SHAP importance rankings, segment profiles |
| **Forecast** | 7-day productivity forecast with confidence bands |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**[Your Name]**
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

*Built as a senior-level data science portfolio project demonstrating end-to-end ML,
behavioral analytics, explainability, and time series forecasting.*
