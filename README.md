# Credit Risk Scoring Model
### Python | scikit-learn | XGBoost | Tableau

---

## Project Overview
Built a binary classification model on 150,000 loan applicants to predict default probability — replicating the same logic banks use for loan approvals. Developed a 300–850 credit scorecard aligned with CIBIL methodology.

---

## Problem Statement
Banks face two risks:
- Approving loans to customers who will default → financial loss
- Rejecting good customers → lost business

This model predicts default probability for each applicant and converts it into an interpretable credit score (300–850), enabling data-driven loan decisions.

---

## Dataset
- **Source:** Kaggle — Give Me Some Credit
- **Size:** 150,000 loan applicants, 11 features
- **Target:** `SeriousDlqin2yrs` (1 = defaulted, 0 = no default)
- **Class Imbalance:** ~93% non-defaulters, ~7% defaulters

---

## Project Phases

### Phase 1 — Exploratory Data Analysis
- Identified missing values: MonthlyIncome (~20%), NumberOfDependents (~2.6%)
- Detected outliers: RevolvingUtilization max = 50,708 (impossible), age = 0
- Confirmed class imbalance: 93% vs 7%
- Correlation analysis: late payment columns most predictive of default

### Phase 2 — Data Cleaning
- Removed age = 0 rows (impossible values)
- Filled MonthlyIncome nulls with median
- Filled NumberOfDependents nulls with 0
- Capped outliers: RevolvingUtilization at 1.0, MonthlyIncome at 99th percentile

### Phase 3 — Feature Engineering

| Feature | Logic |
|---|---|
| `debt_to_income` | DebtRatio × MonthlyIncome |
| `total_late_payments` | Sum of all 3 late payment columns |
| `credit_utilization_risk` | RevolvingUtilization bucketed into Low / Medium / High / Very High |

### Phase 4 — Model Building
- Handled class imbalance using **SMOTE** (oversampling minority class)
- Trained **Logistic Regression** (baseline)
- Trained **XGBoost** (final model)

### Phase 5 — Evaluation

| Model | ROC-AUC | Gini |
|---|---|---|
| Logistic Regression | 0.777 | 0.555 |
| XGBoost | 0.835 | 0.669 |

XGBoost outperformed Logistic Regression across all metrics.

### Phase 6 — Credit Scorecard
Converted default probability to 300–850 score using log-odds transformation:

```
score = offset - factor × log(p / (1 - p))
```

- Score range achieved: 364 – 711
- Median score: 569
- Higher score = lower default risk (aligned with CIBIL methodology)

### Phase 7 — Tableau Dashboard
4-view interactive dashboard:
- Credit Score Distribution (300–850)
- Class Imbalance visualization
- Default Probability vs Credit Score scatter plot
- Average Credit Score by Age Group

---

## Results
- **ROC-AUC: 0.835** | **Gini: 0.669**
- Credit scores ranging 364–711 with median 569
- XGBoost correctly identified high-risk applicants with low credit scores

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | EDA visualizations |
| scikit-learn | Model building & evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| XGBoost | Gradient boosting model |
| Tableau Public | Interactive dashboard |

---

## Files

| File | Description |
|---|---|
| `Credit_Risk_Scoring_Dashboard.jpeg` | Dashboard |
| `CreditRiskModel.ipynb` | Complete notebook — EDA to Scorecard |
| `scorecard_output.csv` | Test set with predicted probability & credit scores |
| `README.md` | Project documentation |


---

## Key Learnings
- Class imbalance handling is critical — without SMOTE, model ignores minority class
- XGBoost handles non-linear patterns better than Logistic Regression
- AUC is more meaningful than accuracy for imbalanced datasets
- Log-odds transformation enables interpretable credit scoring aligned with industry standards

---
