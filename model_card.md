# Model Card — Human Performance Intelligence System

> This document follows the [Model Card](https://arxiv.org/abs/1810.03993) framework
> proposed by Mitchell et al. (2019). Model cards are technical documentation that
> describe a model's intended use, performance, limitations, and ethical considerations.
> They are standard practice in production ML and increasingly expected in senior-level portfolios.

---

## 1. Model Overview

| Property | Details |
|---|---|
| **Project** | Human Performance Intelligence System |
| **Version** | 1.0 |
| **Date** | 2024 |
| **Author** | [Your Name] |
| **License** | MIT |

### Models in this project

| Notebook | Model | Task | Algorithm |
|---|---|---|---|
| 03 | Habit Completion Predictor | Binary classification | XGBoost |
| 03 | Habit Completion Baseline | Binary classification | Logistic Regression |
| 04 | Burnout Risk Predictor | Binary classification | LightGBM |
| 04 | Burnout Risk Baseline | Binary classification | Logistic Regression |
| 04 | BRI Composite Score | Scoring / Rule-based | Weighted formula |
| 05 | Productivity Forecaster | Time series regression | Prophet |

---

## 2. Intended Use

### Primary intended use
- **Research and personal productivity analytics** — understanding behavioral drivers
  of goal achievement and burnout risk in educational or workplace settings
- **Dashboard reporting** — weekly BRI scores for self-monitoring or team wellness programs
- **Prototype system** — demonstrating an end-to-end behavioral data science pipeline

### Intended users
- Data scientists and ML engineers reviewing the methodology
- Product managers evaluating behavioral analytics approaches
- Researchers studying habit formation and burnout

### Out-of-scope uses
- ⚠️ **Clinical mental health diagnosis or treatment** — the BRI is not a validated
  clinical instrument. It should not replace professional mental health assessment.
- ⚠️ **High-stakes employment decisions** — do not use model outputs to make hiring,
  firing, or performance review decisions without proper validation and human oversight.
- ⚠️ **Real-time medical monitoring** — the models are not validated for medical use.

---

## 3. Training Data

### Source
Both datasets are **synthetically generated** to mirror real-world behavioral data.
They are not collected from real individuals.

| Dataset | Rows | Features | Target |
|---|---|---|---|
| Habitica (habit logs) | ~50,000 | 20 | `completed` (binary) |
| StudentLife (sensor) | ~60,000 | 17 | `high_burnout_next_week` (binary) |

### Data characteristics
- **Habitica:** 3,000 synthetic users, habit logs from 2021–2023, 8 habit categories
- **StudentLife:** 500 synthetic students over an 18-week semester
- Both datasets contain **realistic noise**: missing values (3–10%), duplicates (2–3%),
  outliers, mixed date formats, and categorical typos — all documented and cleaned
  in `01_data_pipeline.ipynb`

### Known data limitations
- Synthetic generation means real-world behavioral distributions may differ
- No geographic, cultural, or demographic diversity modeling was applied
- Academic semester structure may not generalize to corporate or freelance workers

---

## 4. Model Performance

### Habit Completion Predictor (XGBoost)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| CV ROC-AUC (5-fold) | ~0.87 | ~0.99 |
| Test ROC-AUC | ~0.87 | ~0.99 |
| Test F1 (weighted) | ~0.84 | ~0.97 |

> ⚠️ **Important caveat:** The high AUC (~0.99) is partly driven by `xp_efficiency`,
> which is computed from the same time window as the target. In a production system,
> this feature must be **lagged by at least 1 day** to prevent look-ahead leakage.
> This limitation is acknowledged and flagged in the notebook.

### Burnout Risk Predictor (LightGBM)

| Metric | Logistic Regression | LightGBM |
|---|---|---|
| Test ROC-AUC | ~0.72 | ~0.82–0.88 |
| Test F1 (weighted) | ~0.74 | ~0.80–0.85 |

> This model uses a **temporal train/test split** (train: weeks 1–14, test: weeks 15–18)
> which is more conservative and realistic than a random split.

### Burnout Risk Index (BRI) — Composite Score

| Component | Weight | Rationale |
|---|---|---|
| Sleep debt | 25% | Primary physiological burnout predictor (literature-backed) |
| Sleep hours | 10% | Total sleep duration (combined sleep signal = 35%) |
| Avg stress | 20% | Chronic psychological stressor |
| Stress-mood ratio | 10% | Emotional dysregulation signal (combined stress signal = 30%) |
| Focus minutes | 12% | Behavioral disengagement indicator |
| Mood score | 8% | Hedonic wellbeing (combined engagement = 20%) |
| Caffeine drinks | 8% | Stress-coping dependency signal |
| Social hours | 7% | Social isolation (combined lifestyle = 15%) |

**Validation:** BRI monotonically predicts goal completion rate across 10-point buckets,
confirming it captures meaningful variance. Scores increase significantly during
deadline weeks (+8–12 BRI points on average).

---

## 5. Evaluation Methodology

### Train/test splits
- **Habit model (NB03):** stratified random 80/20 split + 5-fold cross-validation
- **Burnout model (NB04):** temporal split — weeks 1–14 train, weeks 15–18 test
  *(temporal split used because random splitting on time series = data leakage)*

### Metrics chosen and why
- **ROC-AUC** — robust to class imbalance; measures ranking quality
- **F1 (weighted)** — balances precision and recall; appropriate for imbalanced classes
- **Precision-Recall curve** — explicitly examines the minority class tradeoff
- **Threshold analysis** — optimal threshold tuned for max F1, not hardcoded at 0.5

### Class imbalance handling
- Habit model: SMOTE oversampling on training data + `scale_pos_weight` in XGBoost
- Burnout model: `is_unbalance=True` in LightGBM + early stopping

---

## 6. Data Leakage Mitigations

Leakage occurs when information from the future or from the target variable itself
enters the training features, producing inflated metrics that don't generalize.

| Issue | Detection | Mitigation |
|---|---|---|
| `Complain` column (NB01) | Pearson r ≈ 0.99 with target | **Dropped entirely** |
| `xp_efficiency` look-ahead (NB03) | Same-period calculation | **Flagged; must be lagged 1 day in production** |
| `xp_earned` (NB03) | Only earned on completed days | **Excluded from features** |
| `rolling_completion_7d` (NB03) | Computed from target column | **Excluded from features** |
| Temporal leakage (NB04) | Random split leaks future weeks | **Temporal split enforced** |
| Scaler fit on full data | Would see test distribution | **Scaler fit inside Pipeline on train only** |

---

## 7. Explainability

All models are explained using **SHAP (SHapley Additive exPlanations)**:

- **Global explanations** — beeswarm and bar charts showing top drivers across all predictions
- **Dependence plots** — how individual feature values affect predictions
- **Local explanations** — waterfall plots for individual predictions (high confidence, low confidence, borderline)
- **Behavioral segmentation** — K-Means clustering on SHAP value profiles to identify user archetypes

SHAP was chosen over native feature importance (gain/cover) because it is:
- Consistent (satisfies the efficiency and symmetry axioms)
- Not biased toward high-cardinality features
- Interpretable at both global and instance levels

---

## 8. Ethical Considerations

### Privacy
- No real user data is used. All data is synthetically generated.
- In a real deployment, habit and sensor data is highly sensitive personal information
  and must be handled under GDPR / CCPA / relevant data protection regulations.

### Fairness
- No subgroup fairness analysis (e.g. by age, gender) was performed on model outputs.
- The models include `age` and `gender` as features. In a production system, these
  should be audited for disparate impact across demographic groups before deployment.
- The BRI weights are heuristic and literature-informed, not clinically validated.
  Different weights may be more appropriate for different populations.

### Potential harms
- **Misuse as surveillance:** BRI scores should never be used by employers to monitor
  or penalize employees without explicit informed consent.
- **False reassurance:** A low BRI score does not mean a person is not at risk.
  The model has false negatives and should not replace human judgment.
- **Labeling and stigma:** Classifying someone as "high burnout risk" carries real
  psychological weight. Any system surfacing these scores to users should be
  designed with care, transparency, and opt-in consent.

---

## 9. Limitations Summary

| Limitation | Impact | Recommendation |
|---|---|---|
| Synthetic data | Unknown real-world generalization | Validate on real behavioral dataset before deployment |
| `xp_efficiency` leakage | Inflated NB03 AUC | Lag feature by 1 day in production |
| BRI weights not clinically validated | Scores are indicative, not diagnostic | Conduct validation study with occupational psychologists |
| No demographic fairness audit | Potential disparate impact | Run fairness evaluation before any production use |
| Single semester data (18 weeks) | Limited temporal scope | Retrain on multi-year data for robustness |
| No real-time pipeline | Models run in batch | Build streaming feature store for live scoring |

---

## 10. How to Cite

```
[Your Name] (2024). Human Performance Intelligence System.
GitHub: https://github.com/YOUR_USERNAME/human-performance-intelligence
```

---

*This model card was written as part of responsible ML documentation practices.
Questions or feedback: [your email or LinkedIn]*
